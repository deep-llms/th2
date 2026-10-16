"""Model definitions.

baseline : torchvision ResNet-18 (random init), 3 classes.
forensics: compact residual CNN with a forensics front end:
             - fixed SRM high-pass filters (steganalysis residuals; the 3-kernel
               set used in RGB-N, Zhou et al. CVPR 2018), applied per RGB channel
             - Bayar & Stamm constrained conv (center=-1, off-center sums to 1),
               re-projected every forward pass
             - plain RGB stem
           all at native resolution (stride 1), fused then downsampled through
           ResNet basic blocks. Fully convolutional + GAP, so it accepts any
           input size (train on 256 crops, eval on full 512).

Published from-scratch methods (architecture comparison, 2026-10-11):
resnet18_nodown: ResNet-18 with stride 1 in conv1 and the max-pool, i.e. no early
           downsampling (Gragnaniello et al. ICME 2021 `resnet50nodown`, stride0=1;
           ResNet-50 itself is 23.5M params, over the limit, so ResNet-18).
npr      : NPR (Tan et al. CVPR 2024, official networks/resnet.py): residual
           x - up(down(x)) (nearest, x2), scaled by 2/3, into a ResNet-50 stem +
           layer1 + layer2 (Bottleneck), GAP, linear. The official class also
           allocates the unused layer3/layer4; they are omitted here (identical
           forward pass, but they would count as trainable parameters).
attnet   : GAN attribution net of Yu et al. ICCV 2019 (official classifier/
           networks.py C_patch, fmap_base 1024, fmap_max 512, mode postpool):
           per resolution two 3x3 convs (lrelu 0.2, reflect padding) + 2x2 avg
           pool, terminal 3x3 conv + two 1x1 convs at 4x4. The official config
           uses latent_res=-1 (dense head, fixed input size); we use the code's
           latent_res=4 patch head and average the logit map so the net accepts
           256 crops and full 512 images. He init instead of ProGAN wscale.

All trainable-parameter counts are asserted <= 12,300,000 (contest limit).
"""
import copy
import math
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import Bottleneck

PARAM_LIMIT = 12_300_000


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Forensics front-end layers
# --------------------------------------------------------------------------

# The standard 3-kernel SRM set (residual filters from Fridrich & Kodovsky's
# steganalysis rich models, popularized for forgery detection in RGB-N):
#   q=4: 2nd-order 3x3 laplacian-like; q=12: 5x5 "KB"; q=2: 1st-order 1x3.
_SRM_KERNELS = [
    (
        4.0,
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
    ),
    (
        12.0,
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
    ),
    (
        2.0,
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ),
]


class SRMConv(nn.Module):
    """Fixed high-pass residual filters, each applied to each RGB channel.

    Input (B,3,H,W) -> output (B,9,H,W). Weights are buffers (not trainable,
    so they do not count against the contest parameter limit).
    """

    def __init__(self):
        super().__init__()
        w = torch.zeros(9, 3, 5, 5)
        for f, (q, k) in enumerate(_SRM_KERNELS):
            k = torch.tensor(k, dtype=torch.float32) / q
            for c in range(3):
                w[f * 3 + c, c] = k
        self.register_buffer("weight", w)

    def forward(self, x):
        # fp32 under autocast: bf16 input rounding (~2^-8 relative) is on the
        # order of one uint8 pixel step for values near +/-1, i.e. comparable
        # to the residual signal itself. Compute residuals at full precision.
        with torch.autocast("cuda", enabled=False):
            return F.conv2d(x.float(), self.weight, padding=2)


class BayarConv2d(nn.Module):
    """Constrained conv (Bayar & Stamm 2018): for every (out,in) kernel slice,
    the center weight is -1 and the off-center weights sum to +1, so each
    filter is forced to compute a prediction-error residual and cannot learn
    image content.

    The paper (TIFS 2018, Algorithm 1) enforces the constraint by projecting
    weights after each SGD step; we use the equivalent differentiable
    reparameterization applied every forward (as in MVSS-Net's BayarConv),
    which guarantees the constraint holds exactly at all times. The original
    is grayscale (per-filter constraint); per-(out,in)-slice is the standard
    multi-channel generalization.
    """

    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        assert kernel_size % 2 == 1
        self.k = kernel_size
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

    def _constrained_weight(self):
        c = self.k // 2
        w = self.weight.clone()
        w[:, :, c, c] = 0.0
        # Normalize off-center weights of each (out,in) slice to sum to 1.
        s = w.sum(dim=(2, 3), keepdim=True)
        # Guard against tiny sums blowing up the division (sign-preserving,
        # so a slice whose sum crosses zero does not flip discontinuously).
        s = torch.where(s.abs() < 1e-6, torch.copysign(torch.full_like(s, 1e-6), s), s)
        w = w / s
        # Center = -1: filter output is (predicted value - actual value).
        mask = torch.zeros_like(w)
        mask[:, :, c, c] = 1.0
        return w - mask

    def forward(self, x):
        # fp32 under autocast, same rationale as SRMConv.
        with torch.autocast("cuda", enabled=False):
            return F.conv2d(x.float(), self._constrained_weight().float(), padding=self.k // 2)


class NPRExtract(nn.Module):
    """Neighboring Pixel Relationships (Tan et al., CVPR 2024).

    Residual between the image and its 2x down/up resampled copy; parameter
    free. Diffusion/GAN decoders build images by upsampling, which leaves
    correlations between neighboring pixels that this residual exposes.
    """

    def forward(self, x):
        down = F.interpolate(x, scale_factor=0.5, mode="nearest")
        up = F.interpolate(down, scale_factor=2.0, mode="nearest")
        return x - up


class FADBranch(nn.Module):
    """Frequency-aware decomposition in the spirit of F3-Net (Qian et al. 2020).

    Full-image orthonormal DCT-II -> fixed band masks -> inverse DCT, giving
    one band-limited image per band (4 bands x 3 channels = 12 maps). DCT
    matrices are cached per size; computed in fp32 under autocast for
    numerical stability. No trainable parameters here.
    """

    # Band edges as fractions of the (i+j) diagonal frequency index, matching
    # the reference F3-Net implementation (yyk-wew/F3Net): low = [0, N/2.82),
    # mid = [N/2.82, N/2), high = [N/2, 2N], plus a full-spectrum band. With
    # diag normalized by 2N these become the fractions below. Deviation from
    # F3-Net: no learnable sigmoid mask (ours must handle variable input
    # sizes: 256 crops at train time, full 512 at eval).
    BANDS = [(0.0, 1 / 5.64), (1 / 5.64, 1 / 4), (1 / 4, 1.01), (0.0, 1.01)]

    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _dct_matrix(n, device, dtype):
        k = torch.arange(n, device=device, dtype=dtype)
        d = torch.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
        d = d * np.sqrt(2.0 / n)
        d[0] = d[0] / np.sqrt(2.0)
        return d  # (n,n), orthonormal: D @ D.T = I

    def _get(self, n, device):
        key = (n, device)
        if key not in self._cache:
            d = self._dct_matrix(n, device, torch.float32)
            masks = []
            i = torch.arange(n, device=device, dtype=torch.float32)
            diag = (i[:, None] + i[None, :]) / (2 * n)  # in [0, 1)
            for lo, hi in self.BANDS:
                masks.append(((diag >= lo) & (diag < hi)).float())
            self._cache[key] = (d, torch.stack(masks))  # (n,n), (4,n,n)
        return self._cache[key]

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            b, c, h, w = x.shape
            assert h == w, "FADBranch assumes square inputs"
            d, masks = self._get(h, x.device)
            X = d @ x @ d.T                                   # DCT
            Xb = X.unsqueeze(2) * masks[None, None]           # (B,C,4,H,W)
            bands = d.T @ Xb @ d                              # inverse DCT
            return bands.reshape(b, c * len(self.BANDS), h, w)


# --------------------------------------------------------------------------
# Residual blocks
# --------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


def _stage(in_ch, out_ch, blocks, stride):
    layers = [BasicBlock(in_ch, out_ch, stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_ch, out_ch, 1))
    return nn.Sequential(*layers)


class BlurPool(nn.Module):
    """Anti-aliased downsampling (Zhang, ICML 2019): fixed depthwise [1,2,1]x[1,2,1] binomial
    low-pass filter followed by subsampling. Reflect padding via slices so backward is
    deterministic."""

    def __init__(self, channels, stride=2):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :]) / 16.0
        self.register_buffer("kernel", k[None, None].repeat(channels, 1, 1, 1))
        self.stride, self.channels = stride, channels

    def forward(self, x):
        x = torch.cat([x[..., :, 1:2], x, x[..., :, -2:-1]], dim=-1)
        x = torch.cat([x[..., 1:2, :], x, x[..., -2:-1, :]], dim=-2)
        return F.conv2d(x, self.kernel.to(x.dtype), stride=self.stride, groups=self.channels)


class GeM(nn.Module):
    """Generalized-mean pooling (Radenovic et al., TPAMI 2018) with learnable p (init 3)."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = self.p.float().clamp(min=1.0)
        return x.float().clamp(min=self.eps).pow(p).mean(dim=(2, 3)).pow(1.0 / p)


class AABasicBlock(BasicBlock):
    """BasicBlock whose stride-2 downsampling is anti-aliased as in antialiased-cnns:
    conv(stride 1) -> BN -> ReLU -> BlurPool(stride 2); shortcut: BlurPool -> 1x1 conv."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(in_ch, out_ch, 1)
        self.stride = stride
        if stride != 1:
            self.blur = BlurPool(out_ch, stride)
            self.short = nn.Sequential(BlurPool(in_ch, stride), nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
                                       nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.stride != 1:
            out = self.blur(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


def _stage_aa(in_ch, out_ch, blocks, stride):
    layers = [AABasicBlock(in_ch, out_ch, stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_ch, out_ch, 1))
    return nn.Sequential(*layers)


class PoolBasicBlock(BasicBlock):
    """Downsampling BasicBlock with a pooling layer instead of the stride-2 conv:
    mode "mp"  : conv(s1)-BN-ReLU -> MaxPool 2x2 stride 2 (VGG-style); shortcut AvgPool 2x2 -> 1x1 conv
    mode "mbp" : conv(s1)-BN-ReLU -> MaxPool 2x2 stride 1 -> BlurPool stride 2 (MaxBlurPool, Zhang 2019);
                 shortcut BlurPool -> 1x1 conv."""

    def __init__(self, in_ch, out_ch, stride=1, mode="mp"):
        super().__init__(in_ch, out_ch, 1)
        self.mode, self.stride = mode, stride
        if stride != 1:
            if mode == "mp":
                self.pool = nn.MaxPool2d(2, 2)
                down = nn.AvgPool2d(2, 2)
            else:
                self.pool = nn.Sequential(nn.MaxPool2d(2, 1), BlurPool(out_ch, 2))
                down = BlurPool(in_ch, 2)
            self.short = nn.Sequential(down, nn.Conv2d(in_ch, out_ch, 1, 1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.stride != 1:
            out = self.pool(out)
        out = self.bn2(self.conv2(out))
        # odd input sizes: the mbp path can be one pixel off the shortcut; align (no-op for even sizes)
        h, w = min(out.shape[2], identity.shape[2]), min(out.shape[3], identity.shape[3])
        return F.relu(out[..., :h, :w] + identity[..., :h, :w], inplace=True)


def _stage_pool(mode):
    def make(in_ch, out_ch, blocks, stride):
        layers = [PoolBasicBlock(in_ch, out_ch, stride, mode)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    return make


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------


class ForensicsNet(nn.Module):
    """SRM + Bayar + RGB stems at native resolution -> fused -> residual stages."""

    def __init__(self, num_classes=3, widths=(64, 96, 192, 320, 384), blocks=(2, 2, 2, 2),
                 use_srm=True, use_bayar=True, use_npr=False, use_fad=False, antialias=False, gem=False,
                 pool=None):
        super().__init__()
        stem_w = widths[0]
        self.use_srm = use_srm
        self.use_bayar = use_bayar
        self.use_npr = use_npr
        self.use_fad = use_fad
        fuse_in = 0
        if use_srm:
            self.srm = SRMConv()
            self.srm_bn = nn.BatchNorm2d(9)
            fuse_in += 9
        if use_bayar:
            self.bayar = BayarConv2d(3, 24, 5)
            self.bayar_bn = nn.BatchNorm2d(24)
            fuse_in += 24
        if use_npr:
            self.npr = NPRExtract()
            self.npr_stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
            )
            fuse_in += 16
        if use_fad:
            self.fad = FADBranch()
            self.fad_stem = nn.Sequential(
                nn.Conv2d(12, 24, 3, 1, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True)
            )
            fuse_in += 24
        # When ablating the forensics branches, widen the RGB stem so parameter
        # count stays comparable.
        rgb_w = 24 if (use_srm or use_bayar) else 56
        self.rgb = nn.Sequential(
            nn.Conv2d(3, rgb_w, 3, 1, 1, bias=False), nn.BatchNorm2d(rgb_w), nn.ReLU(inplace=True)
        )
        fuse_in += rgb_w
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, stem_w, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_w),
            nn.ReLU(inplace=True),
        )
        st = _stage_pool(pool) if pool else (_stage_aa if antialias else _stage)
        self.stage1 = st(stem_w, widths[1], blocks[0], stride=2)
        self.stage2 = st(widths[1], widths[2], blocks[1], stride=2)
        self.stage3 = st(widths[2], widths[3], blocks[2], stride=2)
        self.stage4 = st(widths[3], widths[4], blocks[3], stride=2)
        self.pool = GeM() if gem else None
        self.head = nn.Linear(widths[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of each block (standard residual trick).
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        feats = [self.rgb(x)]
        if self.use_srm:
            feats.append(self.srm_bn(self.srm(x)))
        if self.use_bayar:
            feats.append(self.bayar_bn(self.bayar(x)))
        if self.use_npr:
            feats.append(self.npr_stem(self.npr(x)))
        if self.use_fad:
            feats.append(self.fad_stem(self.fad(x)))
        # Residual branches run in fp32; cast everything to the RGB stem's
        # (autocast) dtype so the concat is homogeneous.
        feats = [f.to(feats[0].dtype) for f in feats]
        f = self.fuse(torch.cat(feats, dim=1))
        f = self.stage4(self.stage3(self.stage2(self.stage1(f))))
        f = F.adaptive_avg_pool2d(f, 1).flatten(1) if self.pool is None else self.pool(f)
        return self.head(f)


class NPRNet(nn.Module):
    """NPR detector (Tan et al. CVPR 2024), ported from the official networks/resnet.py."""

    def __init__(self, num_classes=3, layers=(3, 4)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def interpolate(img, factor):
        # Upsample back to the exact input size: identical to the official
        # scale_factor=1/factor call for even sizes (all training crops), and also defined
        # for odd sizes (e.g. post-processed evaluation images), where the official code fails.
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode="nearest", recompute_scale_factor=True),
            size=img.shape[-2:], mode="nearest",
        )

    def forward(self, x):
        npr = x - self.interpolate(x, 0.5)
        x = self.relu(self.bn1(self.conv1(npr * 2.0 / 3.0)))
        x = self.layer2(self.layer1(self.maxpool(x)))
        return self.fc1(torch.flatten(self.avgpool(x), 1))


class AttNet(nn.Module):
    """Yu et al. ICCV 2019 attribution classifier (official C_patch, postpool mode),
    with the code's latent_res=4 patch head; the 4x4-per-crop logit map is averaged."""

    def __init__(self, num_classes=3, resolution=256, fmap_base=1024, fmap_max=512):
        super().__init__()
        nf = lambda s: min(int(fmap_base / 2.0 ** s), fmap_max)
        conv = lambda i, o, k: nn.Conv2d(i, o, k, 1, k // 2, padding_mode="reflect")
        act = lambda: nn.LeakyReLU(0.2, inplace=True)
        layers, ch = [], 3
        for res in range(int(np.log2(resolution)), 2, -1):  # non-terminal blocks, down to 8 -> 4
            layers += [conv(ch, nf(res - 1), 3), act(), conv(nf(res - 1), nf(res - 2), 3), act(),
                       nn.AvgPool2d(2)]
            ch = nf(res - 2)
        layers += [conv(ch, nf(1), 3), act(), conv(nf(1), max(nf(1), num_classes), 1), act(),
                   conv(max(nf(1), num_classes), num_classes, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.net[-1].weight, nonlinearity="linear")  # output layer, gain 1

    def forward(self, x):
        return self.net(x).mean(dim=(2, 3))


class FreqNet(nn.Module):
    """FreqNet (Tan et al. AAAI 2024), ported from the official networks/freqnet.py.

    Changes: no hard-coded .cuda(), 3-way output, and the whole forward runs in fp32
    (torch.fft does not support bf16; the official code trains in fp32).
    """

    def __init__(self, num_classes=3, layers=(3, 4)):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)))
        self.bias1 = nn.Parameter(torch.randn((64,)))
        self.realconv1 = nn.Conv2d(64, 64, 1, bias=False)
        self.imagconv1 = nn.Conv2d(64, 64, 1, bias=False)
        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)))
        self.bias2 = nn.Parameter(torch.randn((64,)))
        self.realconv2 = nn.Conv2d(64, 64, 1, bias=False)
        self.imagconv2 = nn.Conv2d(64, 64, 1, bias=False)
        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias3 = nn.Parameter(torch.randn((256,)))
        self.realconv3 = nn.Conv2d(256, 256, 1, bias=False)
        self.imagconv3 = nn.Conv2d(256, 256, 1, bias=False)
        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias4 = nn.Parameter(torch.randn((256,)))
        self.realconv4 = nn.Conv2d(256, 256, 1, bias=False)
        self.imagconv4 = nn.Conv2d(256, 256, 1, bias=False)
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = NPRNet._make_layer(self, 64, layers[0])
        self.layer2 = NPRNet._make_layer(self, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def hfreqWH(x, scale):
        x = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=[-2, -1])
        b, c, h, w = x.shape
        x[:, :, h // 2 - h // scale:h // 2 + h // scale, w // 2 - w // scale:w // 2 + w // scale] = 0.0
        x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-2, -1]), norm="ortho")
        return F.relu(torch.real(x))

    @staticmethod
    def hfreqC(x, scale):
        x = torch.fft.fftshift(torch.fft.fft(x, dim=1, norm="ortho"), dim=1)
        b, c, h, w = x.shape
        x[:, c // 2 - c // scale:c // 2 + c // scale, :, :] = 0.0
        x = torch.fft.ifft(torch.fft.ifftshift(x, dim=1), dim=1, norm="ortho")
        return F.relu(torch.real(x))

    @staticmethod
    def fcl(x, realconv, imagconv):
        x = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=[-2, -1])
        x = torch.complex(realconv(x.real), imagconv(x.imag))
        x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-2, -1]), norm="ortho")
        return F.relu(torch.real(x))

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            x = x.float().contiguous()
            x = F.relu(F.conv2d(self.hfreqWH(x, 4), self.weight1, self.bias1))
            x = self.fcl(self.hfreqC(x, 4), self.realconv1, self.imagconv1)
            x = F.relu(F.conv2d(self.hfreqWH(x, 4), self.weight2, self.bias2, stride=2))
            x = self.fcl(self.hfreqC(x, 4), self.realconv2, self.imagconv2)
            x = self.layer1(self.maxpool(x))
            x = F.relu(F.conv2d(self.hfreqWH(x, 4), self.weight3, self.bias3))
            x = self.fcl(x, self.realconv3, self.imagconv3)
            x = F.relu(F.conv2d(self.hfreqWH(x, 4), self.weight4, self.bias4, stride=2))
            x = self.fcl(x, self.realconv4, self.imagconv4)
            x = self.layer2(x)
            return self.fc1(torch.flatten(self.avgpool(x), 1))


class WSConv2d(nn.Module):
    """ProGAN equalized-learning-rate conv (N(0,1) weights scaled by gain/sqrt(fan_in) at
    runtime), reflect padding, zero-init bias — as in GANFingerprints networks.conv2d."""

    def __init__(self, cin, cout, k, gain=np.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.scale = gain / np.sqrt(cin * k * k)
        self.pad = k // 2

    def forward(self, x):
        if self.pad:
            x = reflect_pad(x, self.pad)
        return F.conv2d(x, self.weight * self.scale, self.bias)


def reflect_pad(x, p):
    """Same values as F.pad(x, [p]*4, mode="reflect"), built from slices/flip/cat so the
    backward pass is deterministic (reflection_pad2d_backward_cuda is not)."""
    x = torch.cat([x[..., :, 1:p + 1].flip(-1), x, x[..., :, -p - 1:-1].flip(-1)], dim=-1)
    return torch.cat([x[..., 1:p + 1, :].flip(-2), x, x[..., -p - 1:-1, :].flip(-2)], dim=-2)


class WSLinear(nn.Module):
    def __init__(self, fin, fout, gain=np.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin))
        self.bias = nn.Parameter(torch.zeros(fout))
        self.scale = gain / np.sqrt(fin)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class AttNetOrig(nn.Module):
    """Yu et al. ICCV 2019 classifier exactly as the official config (C_patch, fmap_base
    1024, fmap_max 512, latent_res -1 = dense head, postpool, switching_res 4, equalized
    lr), built for the dataset resolution (512), as the code does automatically."""

    def __init__(self, num_classes=3, resolution=512, fmap_base=1024, fmap_max=512):
        super().__init__()
        nf = lambda s: min(int(fmap_base / 2.0 ** s), fmap_max)
        layers, ch = [], 3
        for res in range(int(np.log2(resolution)), 2, -1):
            layers += [WSConv2d(ch, nf(res - 1), 3), nn.LeakyReLU(0.2),
                       WSConv2d(nf(res - 1), nf(res - 2), 3), nn.LeakyReLU(0.2), nn.AvgPool2d(2)]
            ch = nf(res - 2)
        layers += [WSConv2d(ch, nf(1), 3), nn.LeakyReLU(0.2), nn.Flatten(),
                   WSLinear(nf(1) * 16, max(nf(1), num_classes)), nn.LeakyReLU(0.2),
                   WSLinear(max(nf(1), num_classes), num_classes, gain=1.0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrankCNN(nn.Module):
    """Frank et al. ICML 2020 simple pixel CNN (official build_simple_cnn), built for 512
    inputs via the code's image_size option; Keras glorot-uniform init, zero bias."""

    def __init__(self, num_classes=3, resolution=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1), nn.ReLU(),
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * (resolution // 4) ** 2, num_classes),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class DualBN(nn.Module):
    """Processing-split BatchNorm (P1): two BN sets, b0 for clean views and b1 for processed views, shared
    convolutions (cf. AdvProp auxiliary BN, Xie et al. CVPR 2020; DSBN, Chang et al. CVPR 2019).
    `route` selects the set; inference uses b0."""

    def __init__(self, bn):
        super().__init__()
        self.b0, self.b1, self.route = bn, copy.deepcopy(bn), 0

    def forward(self, x):
        return self.b1(x) if self.route else self.b0(x)


def to_split_bn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DualBN(child))
        else:
            to_split_bn(child)
    return module


def set_bn_route(model, route):
    for m in model.modules():
        if isinstance(m, DualBN):
            m.route = route


def expand_to_split_bn(model, state):
    """Map a plain-BN state dict onto a split-BN model (both BN sets start from the shared one)."""
    own = model.state_dict()
    if set(state) == set(own):
        return state
    return {k: state[k.replace(".b0.", ".").replace(".b1.", ".")] for k in own}


def add_grip_x4_blocks(model):
    """Grow a plain GRIP model with four initially zero residual branches.

    Only new blocks are initialized. Preserve the CPU construction RNG so the
    normal training pipeline keeps the same subsequent sampler/dropout seeds.
    This adapts function-preserving network growth and torchvision's final-BN
    zero initialization; it uses no external weights.
    """
    if len(model.layer4) != 2:
        raise ValueError("x4 expansion requires the plain two-block stage4")
    with torch.random.fork_rng(devices=[]):
        for _ in range(4):
            block = Bottleneck(512, 128)
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            nn.init.zeros_(block.bn3.weight)
            model.layer4.append(block)
    return model


def expand_to_grip_x4(model, state):
    """Map an exact plain-parent state while retaining every new block tensor.

    A complete x4 state is also accepted for subsequent initialization. Any
    other missing/extra key, shape mismatch or dtype mismatch is rejected.
    Merge into a complete state before strict loading, including BN counters;
    this avoids silently accepting unrelated missing keys with strict=False.
    """
    if len(model.layer4) != 6 or not all(isinstance(b, Bottleneck) for b in list(model.layer4)[2:]):
        raise ValueError("Expected a grip_nodown_x4 model")
    own = model.state_dict()
    prefixes = tuple(f"layer4.{i}." for i in range(2, 6))
    new_keys = {k for k in own if k.startswith(prefixes)}
    plain_keys = set(own) - new_keys
    if set(state) not in (plain_keys, set(own)):
        raise ValueError("Parent must contain exactly the plain GRIP state or the complete x4 state")
    for key, value in state.items():
        if value.shape != own[key].shape or value.dtype != own[key].dtype:
            raise ValueError(f"Parent tensor shape/dtype mismatch: {key}")
    if set(state) == set(own):
        return state
    for block in list(model.layer4)[2:]:
        if torch.count_nonzero(block.bn3.weight).item() or torch.count_nonzero(block.bn3.bias).item():
            raise ValueError("New residual branches must be zero-initialized before plain-parent loading")
    return {**own, **state}


class YCbCrStem(nn.Module):
    """JPEG-4:2:0-aware stem (P2): fixed RGB -> YCbCr (BT.601), luma through a stride-1 7x7 conv at full
    resolution, chroma 2x2-average-pooled (the 4:2:0 sampling grid) through its own 7x7 conv at half
    resolution, nearest upsampled (this model's chosen reconstruction rule; real decoders' chroma filters vary)
    and concatenated. A chroma low-pass/decimation prior, not a JPEG invariance guarantee. Expects ImageNet-normalised input."""

    def __init__(self, y_ch=48, c_ch=16):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("m", torch.tensor([[0.299, 0.587, 0.114],
                                                [-0.168736, -0.331264, 0.5],
                                                [0.5, -0.418688, -0.081312]]))
        self.y = nn.Sequential(nn.Conv2d(1, y_ch, 7, 1, 3, bias=False), nn.BatchNorm2d(y_ch), nn.ReLU(inplace=True))
        self.c = nn.Sequential(nn.AvgPool2d(2, 2, ceil_mode=True), nn.Conv2d(2, c_ch, 7, 1, 3, bias=False),
                               nn.BatchNorm2d(c_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        # fixed colour transform in float32 even under autocast (bf16 would leak chroma into neutral greys)
        with torch.autocast(device_type=x.device.type, enabled=False):
            rgb = x.float() * self.std + self.mean
            ycc = torch.einsum("ij,bjhw->bihw", self.m.float(), rgb)
        fy = self.y(ycc[:, :1])
        fc = F.interpolate(self.c(ycc[:, 1:]), size=fy.shape[-2:], mode="nearest")
        return torch.cat([fy, fc], 1)



class RadialWhiten(nn.Module):
    """Per-image radial power-spectrum whitening (report R): un-normalise to RGB, fixed BT.601 YCbCr, remove the mean,
    mirror-pad 2x, divide each channel's spectrum by the square root of its own smoothed radial power profile (nb radial
    bins, normalised frequency so it is size-agnostic), invert and scale to unit std. Removes the image's global spectral
    envelope (sharpness/softness, contrast, saturation gain) while keeping phase and narrow peaks. No parameters."""

    def __init__(self, nb=96, alpha=1.0, eps=1e-6, luma_only=False):
        super().__init__()
        self.nb, self.a, self.eps, self.luma_only = nb, alpha, eps, luma_only
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("m", torch.tensor([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]]))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = torch.einsum("ij,bjhw->bihw", self.m.float(), x.float() * self.std + self.mean)
            B, C, H, W = x.shape
            x = x - x.mean((2, 3), keepdim=True)
            x = torch.cat([x, x.flip(3)], 3)
            x = torch.cat([x, x.flip(2)], 2)
            Z = torch.fft.rfft2(x)
            fy = torch.fft.fftfreq(2 * H, device=x.device)[:, None]
            fx = torch.fft.rfftfreq(2 * W, device=x.device)[None]
            idx = ((fy ** 2 + fx ** 2).sqrt() / 0.7072 * self.nb).long().clamp(max=self.nb - 1).flatten()
            P = Z.abs().pow(2).flatten(2)
            S = torch.zeros(B, C, self.nb, device=x.device).index_add_(2, idx, P)
            S = S / torch.bincount(idx, minlength=self.nb).clamp(min=1)
            S = F.avg_pool1d(S, 3, 1, 1, count_include_pad=False)
            # absolute floor: a constant (flat) crop has S == 0 everywhere and would give inf * 0 = NaN (s53/s54 epoch 22)
            G = (S + self.eps * S.mean(-1, keepdim=True) + 1e-12).pow(-self.a / 2)[..., idx].view_as(Z.real)
            if self.luma_only:   # chroma keeps its own spectrum (4:2:0 test chroma has almost no HF to amplify)
                G = torch.cat([G[:, :1], torch.ones_like(G[:, 1:])], 1)
            Z = Z * G
            Z[..., 0, 0] = 0
            y = torch.fft.irfft2(Z, s=(2 * H, 2 * W))[..., :H, :W]
            return y / y.std((2, 3), keepdim=True).clamp_min(1e-6)



class DSUHook:
    """DSU (Li et al., "Uncertainty Modeling for Out-of-Distribution Generalization", ICLR 2022): during training only,
    with probability p re-sample each instance's channel mean/std from Gaussians whose std is the batch std of those
    statistics. Attached as a forward hook (no parameters, unchanged state-dict keys, identity at eval)."""

    def __init__(self, p=0.5, eps=1e-6):
        self.p, self.eps = p, eps

    def __call__(self, module, inputs, x):
        if not module.training or torch.rand(1).item() > self.p:
            return x
        xf = x.float()
        mu = xf.mean(dim=(2, 3), keepdim=True)
        sig = (xf.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        s_mu = (mu.var(dim=0, keepdim=True) + self.eps).sqrt()
        s_sig = (sig.var(dim=0, keepdim=True) + self.eps).sqrt()
        beta = mu + torch.randn_like(mu) * s_mu
        gamma = sig + torch.randn_like(sig) * s_sig
        return (((xf - mu) / sig) * gamma + beta).to(x.dtype)



def _evidence_forward(self, x):
    """Evidence-based MIL readout (grip_nodown_mil): the shared dropout+linear head scores every location of the final
    feature map; the REAL logit is the mean over locations (an image is real only if it looks real everywhere), each FAKE
    logit is a normalised log-sum-exp (smooth max) over locations (one location with strong generator evidence suffices).
    Same parameters as grip_nodown; the readout is linear in the features, so r -> 0 recovers the GAP model."""
    f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f = self.layer4(self.layer3(self.layer2(self.layer1(f))))          # (B, 512, h, w)
    z = self.fc(f.flatten(2).transpose(1, 2))                          # (B, h*w, 3): dropout + linear per location
    n = z.shape[1]
    r = self.mil_r
    real = z[..., 0].mean(1, keepdim=True)
    fake = (torch.logsumexp(r * z[..., 1:].float(), dim=1) - math.log(n)) / r
    return torch.cat([real.float(), fake], 1).to(z.dtype)



def _relative_evidence_forward(self, x):
    """Relative-evidence MIL readout (grip_nodown_relmil; corrected after the research agent's review):
    per-location logits z_i = W f_i + b from the shared linear head; local fake-vs-real evidence m_ik = z_ik - z_i0
    (invariant to a shift shared by all classes at a location); image logits = [mean z_0, mean z_0 + LME_r(m_1),
    mean z_0 + LME_r(m_2)] with LME_r the normalised log-sum-exp. Dropout uses ONE channel mask per image, broadcast over
    all locations, so r -> 0 reproduces the GAP model (same weights, same dropout distribution) exactly in eval mode."""
    f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f = self.layer4(self.layer3(self.layer2(self.layer1(f))))          # (B, C, h, w)
    drop, lin = self.fc[0], self.fc[1]
    if self.training and drop.p > 0:
        mask = torch.bernoulli(torch.full((f.shape[0], f.shape[1], 1, 1), 1 - drop.p, device=f.device, dtype=f.dtype)) / (1 - drop.p)
        f = f * mask
    z = lin(f.flatten(2).transpose(1, 2)).float()                       # (B, N, 3)
    n, r = z.shape[1], self.mil_r
    base = z[..., 0].mean(1, keepdim=True)
    m = z[..., 1:] - z[..., :1]
    ev = (torch.logsumexp(r * m, dim=1) - math.log(n)) / r              # (B, 2)
    return torch.cat([base, base + ev], 1)


class TruncatedResidualStem(nn.Module):
    """Paper-inspired evidence input (our early-fusion design): next to the normalised RGB, per-channel local residuals
    x - mean_k(x) at k = 3 and 5 (in grey levels, fp32), truncated to +-t grey levels and scaled to [-1, 1] (TLU,
    Ye et al. 2017). The truncation keeps the weak noise-level evidence and discards strong edges/content; a linear
    first convolution cannot compute it. Output: 9 channels for the following convolution. No parameters."""

    def __init__(self, t=3.0):
        super().__init__()
        self.t = t
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            g = (x.float() * self.std + self.mean) * 255.0
            out = [x.float()]
            for k in (3, 5):
                r = g - F.avg_pool2d(g, k, 1, k // 2, count_include_pad=False)
                out.append(r.clamp(-self.t, self.t) / self.t)
            return torch.cat(out, 1)



class IBN(nn.Module):
    """IBN-a split normalisation (Pan et al., ECCV 2018): instance norm (affine) on the first half of the channels,
    batch norm on the rest."""

    def __init__(self, planes):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        a, b = torch.split(x, [self.half, x.shape[1] - self.half], 1)
        return torch.cat([self.IN(a.contiguous()), self.BN(b.contiguous())], 1)



def _ibnb_block_forward(self, x):
    """torchvision BasicBlock forward with IBN-b's instance norm on the residual sum before the final ReLU."""
    identity = x if self.downsample is None else self.downsample(x)
    out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
    return self.relu(self.ibn(out + identity))



def _dense_aux_forward(self, x):
    """grip_nodown_dense (our design, from the flat-region error diagnosis): the ordinary GAP head decides; during training
    a 1x1 3-class head on the layer3 map is supervised at EVERY location with the image label (dense auxiliary loss), so
    flat/low-texture regions must also carry class evidence. The location logits are kept in self.dense_logits; the
    inference output is exactly the GAP head's."""
    f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f3 = self.layer3(self.layer2(self.layer1(f)))
    if self.training:
        self.dense_logits = self.dense_head(f3)                         # (B, 3, h, w)
    return self.fc(torch.flatten(self.avgpool(self.layer4(f3)), 1))



def _texture_cov_forward(self, x):
    """<base>_twcov (our design): texture-weighted first-order deep evidence (as _tw) + FLAT-weighted second-order
    statistics of the full-resolution layer1 features. Diagnosis: in flat regions the only real-vs-fake cue that is stable
    across train/dev/test is the cross-channel correlation of the noise residual (camera demosaicing), which a mean
    (first-order) pool cannot represent. layer1 (64 ch) -> 1x1 conv to 32 ch -> covariance weighted by
    (local gradient energy + eps)^-0.5 -> trace normalisation -> upper triangle (528) -> signed sqrt -> LayerNorm,
    concatenated with the 512-d pooled feature before a 1040->3 linear head (cov part zero-initialised)."""
    f0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f1 = self.layer1(f0)
    f = self.layer4(self.layer3(self.layer2(f1)))
    with torch.autocast(device_type=x.device.type, enabled=False):
        rgb = x.float() * self.tw_std + self.tw_mean
        g = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        gm = (g[:, 1:, :-1] - g[:, :-1, :-1]).abs() + (g[:, :-1, 1:] - g[:, :-1, :-1]).abs()
        gm = F.avg_pool2d(gm[:, None], 8, 8)                              # local texture energy, 8x8 cells
        w = (F.adaptive_avg_pool2d(gm, f.shape[-2:]) + 1e-3).pow(self.tw_gamma)
        w = w / w.sum((2, 3), keepdim=True)
        pooled = (f.float() * w).sum((2, 3))
        r = self.cov_reduce(f1.float())                                   # (B, 32, H, W) at input resolution
        wf = F.interpolate((gm + 1e-3).pow(-0.5), size=r.shape[-2:], mode="nearest")
        wf = wf / wf.sum((2, 3), keepdim=True)
        mu = (r * wf).sum((2, 3), keepdim=True)
        rc = (r - mu) * wf.sqrt()
        B, Cc = rc.shape[:2]
        cov = torch.einsum("bchw,bdhw->bcd", rc, rc)
        cov = cov / cov.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-8)[:, None, None]
        iu = torch.triu_indices(Cc, Cc, device=x.device)
        v = cov[:, iu[0], iu[1]] * Cc
        v = self.cov_norm(v.sign() * (v.abs() + 1e-8).sqrt())
        z = self.fc[1](self.fc[0](torch.cat([pooled, v], 1)))
    return z



def _texture_pool_forward(self, x):
    """Texture-weighted evidence pooling (our design, from the flat-region diagnosis): the GAP of the final map is
    replaced by a weighted mean whose fixed weights are the local luma gradient energy of the INPUT, pooled to the map
    resolution and raised to tw_gamma (0 = plain GAP). Flat regions, which carry little forensic signal after
    (re-)compression, get less say. Parameter-free; identical head."""
    f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f3 = self.layer3(self.layer2(self.layer1(f)))
    if self.training and hasattr(self, "dense_head"):
        self.dense_logits = self.dense_head(f3)
    f = self.layer4(f3)
    with torch.autocast(device_type=x.device.type, enabled=False):
        rgb = x.float() * self.tw_std + self.tw_mean
        g = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        gm = (g[:, 1:, :-1] - g[:, :-1, :-1]).abs() + (g[:, :-1, 1:] - g[:, :-1, :-1]).abs()
        t = F.adaptive_avg_pool2d(gm[:, None], f.shape[-2:])
        w = (t + 1e-3).pow(self.tw_gamma)
        w = w / w.sum((2, 3), keepdim=True)
        pooled = (f.float() * w).sum((2, 3))
    return self.fc(pooled.to(f.dtype))



def _hier_patch_forward(self, x):
    """pi_net_h (paper-inspired, our implementation): pi_net_l trunk; texture-weighted pooled feature -> dropout ->
    realness logit a and generator logits b; a additionally receives a learned multiple (init 0) of the TEXTURE-WEIGHTED
    mean of per-location realness logits from a 1x1 head on the stage-3 map. Hierarchical distribution
    P(real) = sigmoid(a), P(A,B) = (1 - sigmoid(a)) softmax(b), returned as log-probabilities (so CE applies directly).
    The per-location realness map is kept in self.patch_real_map for its BCE supervision during training."""
    f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f3 = self.layer3(self.layer2(self.layer1(f)))
    f = self.layer4(f3)
    with torch.autocast(device_type=x.device.type, enabled=False):
        rgb = x.float() * self.tw_std + self.tw_mean
        g = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        gm = (g[:, 1:, :-1] - g[:, :-1, :-1]).abs() + (g[:, :-1, 1:] - g[:, :-1, :-1]).abs()
        w4 = (F.adaptive_avg_pool2d(gm[:, None], f.shape[-2:]) + 1e-3).pow(self.tw_gamma)
        w4 = w4 / w4.sum((2, 3), keepdim=True)
        pooled = (f.float() * w4).sum((2, 3))
        w3 = (F.adaptive_avg_pool2d(gm[:, None], f3.shape[-2:]) + 1e-3).pow(self.tw_gamma)
        w3 = w3 / w3.sum((2, 3), keepdim=True)
        pm = self.patch_real(f3.float())                                    # (B, 1, h, w) local realness logits
        if self.training:
            self.patch_real_map = pm
        h = self.head_drop(pooled)
        a = self.fc_real(h)[:, 0] + self.patch_scale * (pm * w3).sum((1, 2, 3))
        b = self.fc_gen(h)
        return torch.cat([F.logsigmoid(a)[:, None], F.logsigmoid(-a)[:, None] + F.log_softmax(b, 1)], 1)



def _patchvote_forward(self, x):
    """Patch-level realness vote on a plain GRIP/IBN model (ours): the 3-class GAP head is kept; a 1x1 head on the
    stage-3 map gives per-location real-vs-synthetic logits, robustly aggregated (texture-weighted mean or median) and
    added, times a learned scale (init 0), to the REAL logit. The map is kept for its per-location BCE supervision."""
    f = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f3 = self.layer3(self.layer2(self.layer1(f)))
    z = self.fc(torch.flatten(self.avgpool(self.layer4(f3)), 1)).float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        pm = self.patch_real(f3.float())                                    # (B, 1, h, w)
        if self.training:
            self.patch_real_map = pm
        if self.pv_agg == "median":
            agg = pm.flatten(1).median(1).values
        else:
            rgb = x.float() * self.tw_std + self.tw_mean
            g = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            gm = (g[:, 1:, :-1] - g[:, :-1, :-1]).abs() + (g[:, :-1, 1:] - g[:, :-1, :-1]).abs()
            w = (F.adaptive_avg_pool2d(gm[:, None], pm.shape[-2:]) + 1e-3).pow(0.5)
            agg = (pm * w / w.sum((2, 3), keepdim=True)).sum((1, 2, 3))
        return torch.cat([z[:, :1] + self.patch_scale * agg[:, None], z[:, 1:]], 1)



def _texture_cov_patch_forward(self, x):
    """<base>_twcov_pd: texture-weighted + flat-weighted covariance pooling (as _twcov) PLUS a patch-level realness vote
    (1x1 head on stage 3, texture-weighted mean, added to the REAL logit with a learned scale, init 0) and a dense
    per-location 3-class auxiliary head on stage 3 (training only)."""
    f0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    f1 = self.layer1(f0)
    f3 = self.layer3(self.layer2(f1))
    if self.training:
        self.dense_logits = self.dense_head(f3)
    f = self.layer4(f3)
    with torch.autocast(device_type=x.device.type, enabled=False):
        rgb = x.float() * self.tw_std + self.tw_mean
        g = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        gm = (g[:, 1:, :-1] - g[:, :-1, :-1]).abs() + (g[:, :-1, 1:] - g[:, :-1, :-1]).abs()
        gm = F.avg_pool2d(gm[:, None], 8, 8)
        w = (F.adaptive_avg_pool2d(gm, f.shape[-2:]) + 1e-3).pow(self.tw_gamma)
        w = w / w.sum((2, 3), keepdim=True)
        pooled = (f.float() * w).sum((2, 3))
        r = self.cov_reduce(f1.float())
        wf = F.interpolate((gm + 1e-3).pow(-0.5), size=r.shape[-2:], mode="nearest")
        wf = wf / wf.sum((2, 3), keepdim=True)
        mu = (r * wf).sum((2, 3), keepdim=True)
        rc = (r - mu) * wf.sqrt()
        Cc = rc.shape[1]
        cov = torch.einsum("bchw,bdhw->bcd", rc, rc)
        cov = cov / cov.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-8)[:, None, None]
        iu = torch.triu_indices(Cc, Cc, device=x.device)
        v = cov[:, iu[0], iu[1]] * Cc
        v = self.cov_norm(v.sign() * (v.abs() + 1e-8).sqrt())
        pm = self.patch_real(f3.float())
        if self.training:
            self.patch_real_map = pm
        w3 = (F.adaptive_avg_pool2d(gm, pm.shape[-2:]) + 1e-3).pow(self.tw_gamma)
        agg = (pm * w3 / w3.sum((2, 3), keepdim=True)).sum((1, 2, 3))
        z = self.fc[1](self.fc[0](torch.cat([pooled, v], 1)))
        return torch.cat([z[:, :1] + self.patch_scale * agg[:, None], z[:, 1:]], 1)



def build_model(name, num_classes=3):
    if name in ("srresnet", "srresnet_stats", "srresnet_wide_stats"):
        from deadline_candidates import SRResNet
        model = SRResNet(stats=name.endswith("_stats"), num_classes=num_classes,
                         final_width=536 if name == "srresnet_wide_stats" else 512)
    elif name in ("grip_nodown_stats", "grip_nodown_ibna_stats"):
        from deadline_candidates import MultiDepthStats
        model = MultiDepthStats(build_model(name[:-6], num_classes), num_classes)
    elif name == "baseline":
        model = resnet18(weights=None, num_classes=num_classes)
    elif name == "grip_nodown":
        # Gragnaniello ICME 2021 res50nodown (GRIP training code) scaled to ResNet-18:
        # stride 1 in conv1 and max-pool, dropout 0.5 before the linear head (fc N(0, 0.02)).
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1.stride = (1, 1)
        model.maxpool.stride = 1
        fc = nn.Linear(512, num_classes)
        nn.init.normal_(fc.weight, 0.0, 0.02)
        nn.init.zeros_(fc.bias)
        model.fc = nn.Sequential(nn.Dropout(0.5), fc)
    elif name == "grip_nodown_x4":
        model = add_grip_x4_blocks(build_model("grip_nodown", num_classes))
    elif name == "grip_nodown_relmil":
        model = build_model("grip_nodown", num_classes)
        model.mil_r = 2.0
        model.forward = types.MethodType(_relative_evidence_forward, model)
    elif name == "grip_nodown_mil":
        model = build_model("grip_nodown", num_classes)
        model.mil_r = 2.0
        model.forward = types.MethodType(_evidence_forward, model)
    elif name == "grip_nodown_dsu":
        # DSU after layer2, layer3, layer4 (deeper positions first, as recommended for forensic texture cues)
        model = build_model("grip_nodown", num_classes)
        for layer in (model.layer2, model.layer3, model.layer4):
            layer.register_forward_hook(DSUHook(0.5))
    elif name.endswith("_pvtw") or name.endswith("_pvmed"):
        # <base>_pvtw / <base>_pvmed: patch-level realness vote (texture-weighted mean / median) on a plain GRIP/IBN base
        base, agg = name.rsplit("_", 1)
        model = build_model(base, num_classes)
        model.pv_agg = "median" if agg == "pvmed" else "tw"
        model.patch_real = nn.Conv2d(256, 1, 1)
        model.patch_scale = nn.Parameter(torch.zeros(()))
        model.register_buffer("tw_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        model.register_buffer("tw_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        model.forward = types.MethodType(_patchvote_forward, model)
    elif name == "pi_net_h":
        # pi_net_l trunk + hierarchical realness/generator head + patch-realness evidence fused into the realness logit
        model = build_model("pi_net_l", num_classes)
        del model.dense_head
        model.head_drop = nn.Dropout(0.5)
        model.fc_real = nn.Linear(512, 1)
        model.fc_gen = nn.Linear(512, 2)
        for fc in (model.fc_real, model.fc_gen):
            nn.init.normal_(fc.weight, 0.0, 0.02)
            nn.init.zeros_(fc.bias)
        del model.fc
        model.patch_real = nn.Conv2d(256, 1, 1)
        model.patch_scale = nn.Parameter(torch.zeros(()))
        model.forward = types.MethodType(_hier_patch_forward, model)
    elif name == "pi_net_l":
        # pi_net with the spare parameter budget spent where the forensic evidence lives: +1 BasicBlock in stage 1
        # (64 ch, full resolution) and +2 in stage 2 (128 ch); IBN-b is applied to the (new) last blocks of stages 1-2
        from torchvision.models.resnet import BasicBlock
        model = build_model("grip_nodown", num_classes)
        model.layer1.append(BasicBlock(64, 64))
        model.layer2.append(BasicBlock(128, 128))
        model.layer2.append(BasicBlock(128, 128))
        model.bn1 = nn.InstanceNorm2d(64, affine=True)
        for layer, c in ((model.layer1, 64), (model.layer2, 128)):
            blk = layer[-1]
            blk.ibn = nn.InstanceNorm2d(c, affine=True)
            blk.forward = types.MethodType(_ibnb_block_forward, blk)
        model.tw_gamma = 0.5
        model.register_buffer("tw_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        model.register_buffer("tw_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        model.forward = types.MethodType(_texture_pool_forward, model)
        model.conv1 = nn.Sequential(TruncatedResidualStem(), nn.Conv2d(9, 64, 7, 1, 3, bias=False))
        model.dense_head = nn.Conv2d(256, num_classes, 1)
    elif name == "pi_net":
        # paper-inspired, our own composition: IBN-b no-down ResNet-18 + truncated multi-scale residual input channels
        # (early fusion) + texture-weighted evidence pooling + dense per-location 3-class auxiliary head (training only)
        model = build_model("grip_nodown_ibnb_tw", num_classes)
        model.conv1 = nn.Sequential(TruncatedResidualStem(), nn.Conv2d(9, 64, 7, 1, 3, bias=False))
        model.dense_head = nn.Conv2d(256, num_classes, 1)
    elif name == "grip_nodown_dense":
        model = build_model("grip_nodown", num_classes)
        model.dense_head = nn.Conv2d(256, num_classes, 1)
        model.forward = types.MethodType(_dense_aux_forward, model)
    elif name.endswith("_twcov_pd"):
        # <base>_twcov + patch-level realness vote + dense 3-class auxiliary head (all-components variant)
        model = build_model(name[:-3], num_classes)
        model.dense_head = nn.Conv2d(256, num_classes, 1)
        model.patch_real = nn.Conv2d(256, 1, 1)
        model.patch_scale = nn.Parameter(torch.zeros(()))
        model.forward = types.MethodType(_texture_cov_patch_forward, model)
    elif name.endswith("_twcov"):
        model = build_model(name[:-6] + "_tw", num_classes)
        model.cov_reduce = nn.Conv2d(64, 32, 1, bias=False)
        model.cov_norm = nn.LayerNorm(32 * 33 // 2)
        old_fc = model.fc[1]
        fc = nn.Linear(512 + 32 * 33 // 2, num_classes)
        with torch.no_grad():
            fc.weight.zero_()
            fc.weight[:, :512] = old_fc.weight
            fc.bias.copy_(old_fc.bias)
        model.fc = nn.Sequential(model.fc[0], fc)
        model.forward = types.MethodType(_texture_cov_forward, model)
    elif name.endswith("_tw"):
        # <base>_tw: texture-weighted evidence pooling on top of any GRIP variant (gamma 0.5)
        model = build_model(name[:-3], num_classes)
        model.tw_gamma = 0.5
        model.register_buffer("tw_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        model.register_buffer("tw_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        model.forward = types.MethodType(_texture_pool_forward, model)
    elif name == "grip_nodown_ibna_l":
        # IBN-a with the spare parameter budget in the fine-evidence stages: +1 BasicBlock in stage 1 (full resolution),
        # +2 in stage 2 -> [3, 4, 2, 2]; IBN-a (half IN in bn1) in every block of stages 1-3
        from torchvision.models.resnet import BasicBlock
        model = build_model("grip_nodown", num_classes)
        model.layer1.append(BasicBlock(64, 64))
        model.layer2.append(BasicBlock(128, 128))
        model.layer2.append(BasicBlock(128, 128))
        for layer in (model.layer1, model.layer2, model.layer3):
            for blk in layer:
                blk.bn1 = IBN(blk.bn1.num_features)
    elif name == "grip_nodown_ibna":
        # IBN-Net-a: the first BN of every BasicBlock in layer1-3 becomes IBN (half IN, half BN); stem and layer4 keep BN
        model = build_model("grip_nodown", num_classes)
        for layer in (model.layer1, model.layer2, model.layer3):
            for blk in layer:
                blk.bn1 = IBN(blk.bn1.num_features)
    elif name == "grip_nodown_ibnb3":
        # IBN-b with the residual-sum instance norm extended to the last block of layer3
        model = build_model("grip_nodown_ibnb", num_classes)
        blk = model.layer3[-1]
        blk.ibn = nn.InstanceNorm2d(256, affine=True)
        blk.forward = types.MethodType(_ibnb_block_forward, blk)
    elif name == "grip_nodown_ibnb":
        # IBN-Net-b (Pan et al., ECCV 2018) on the no-down ResNet-18: affine instance norm replaces the stem BN, and is
        # applied to the residual sum of the last block of layer1 and layer2 (per-image gain normalisation of early
        # responses, report R rank 2). +1,536 parameters.
        model = build_model("grip_nodown", num_classes)
        model.bn1 = nn.InstanceNorm2d(64, affine=True)
        for layer, c in ((model.layer1, 64), (model.layer2, 128)):
            blk = layer[-1]
            blk.ibn = nn.InstanceNorm2d(c, affine=True)
            blk.forward = types.MethodType(_ibnb_block_forward, blk)
    elif name in ("grip_nodown_rw", "grip_nodown_rwy"):
        # fixed per-image radial spectral whitening in front of the unchanged no-down ResNet-18 (report R, rank 1);
        # _rwy whitens luma only (chroma only mean-removed and std-scaled)
        model = build_model("grip_nodown", num_classes)
        model.conv1 = nn.Sequential(RadialWhiten(luma_only=name.endswith("rwy")), model.conv1)
    elif name in ("grip_nodown_sbn", "grip_ycc"):
        model = build_model("grip_nodown", num_classes)
        if name == "grip_ycc":
            model.conv1, model.bn1, model.relu = YCbCrStem(), nn.Identity(), nn.Identity()
        else:
            to_split_bn(model)
    elif name in ("grip_nodown_l3", "grip_nodown_l2"):
        # Limited receptive field (Chai et al., ECCV 2020: truncated, patch-level classifiers rely on local
        # artefacts instead of global content): Gragnaniello no-down ResNet-18 truncated after layer3 / layer2.
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1.stride = (1, 1)
        model.maxpool.stride = 1
        model.layer4 = nn.Identity()
        width = 256
        if name.endswith("l2"):
            model.layer3 = nn.Identity()
            width = 128
        fc = nn.Linear(width, num_classes)
        nn.init.normal_(fc.weight, 0.0, 0.02)
        nn.init.zeros_(fc.bias)
        model.fc = nn.Sequential(nn.Dropout(0.5), fc)
    elif name == "freqnet":
        model = FreqNet(num_classes=num_classes)
    elif name == "attnet_orig":
        model = AttNetOrig(num_classes=num_classes)
    elif name == "frank_cnn":
        model = FrankCNN(num_classes=num_classes)
    elif name == "resnet18_nodown":
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1.stride = (1, 1)
        model.maxpool.stride = 1
    elif name == "npr":
        model = NPRNet(num_classes=num_classes)
    elif name == "attnet":
        model = AttNet(num_classes=num_classes)
    elif name == "forensics":
        model = ForensicsNet(num_classes=num_classes)
    elif name == "rgbnet":
        # Ablation: same topology as forensics, no SRM/Bayar branches.
        model = ForensicsNet(num_classes=num_classes, use_srm=False, use_bayar=False)
    elif name.startswith("rgbnet_") and set(name.split("_")[1:]) <= {"aa", "gem", "wide", "deep", "mp", "mbp"} \
            and name not in ("rgbnet_wide",):
        # Our improvements, combinable: anti-aliased downsampling (Zhang 2019), GeM pooling
        # (Radenovic 2018), and use of the remaining parameter budget: "wide" = stage widths
        # 336/448 (12.15M), "deep" = one more block in stage 3 (11.97M).
        parts = set(name.split("_")[1:])
        widths = (64, 96, 192, 336, 448) if "wide" in parts else (64, 96, 192, 320, 384)
        blocks = (2, 2, 3, 2) if "deep" in parts else (2, 2, 2, 2)
        model = ForensicsNet(num_classes=num_classes, use_srm=False, use_bayar=False, widths=widths,
                             blocks=blocks, antialias="aa" in parts, gem="gem" in parts,
                             pool="mbp" if "mbp" in parts else ("mp" if "mp" in parts else None))
    elif name == "rgbnet_wide":
        # rgbnet scaled to use most of the 12.3M parameter budget.
        model = ForensicsNet(num_classes=num_classes, use_srm=False, use_bayar=False,
                             widths=(64, 96, 192, 336, 448))
    elif name == "forensics_npr":
        model = ForensicsNet(num_classes=num_classes, use_npr=True)
    elif name == "forensics_fad":
        model = ForensicsNet(num_classes=num_classes, use_fad=True)
    elif name == "forensics_all":
        model = ForensicsNet(num_classes=num_classes, use_npr=True, use_fad=True)
    else:
        raise ValueError(name)
    n = count_trainable(model)
    assert n <= PARAM_LIMIT, f"{name}: {n:,} params exceeds limit {PARAM_LIMIT:,}"
    return model


if __name__ == "__main__":
    for name in ["baseline", "forensics"]:
        m = build_model(name)
        n = count_trainable(m)
        x = torch.randn(2, 3, 256, 256)
        y = m(x)
        y512 = m(torch.randn(1, 3, 512, 512))
        print(f"{name}: {n:,} trainable params | out {tuple(y.shape)} | 512-in out {tuple(y512.shape)}")

    # Sanity: SRM responds to noise, not to flat images.
    srm = SRMConv()
    flat = torch.full((1, 3, 32, 32), 0.5)
    noisy = flat + torch.randn_like(flat) * 0.1
    print("SRM(flat) max |.|:", srm(flat).abs().max().item())
    print("SRM(noisy) std:", srm(noisy).std().item())

    # Sanity: Bayar constraint holds after projection.
    b = BayarConv2d(3, 8, 5)
    w = b._constrained_weight()
    center = w[:, :, 2, 2]
    off = w.sum(dim=(2, 3)) - center
    print("Bayar center==-1:", torch.allclose(center, torch.full_like(center, -1.0), atol=1e-5))
    print("Bayar off-center sums==1:", torch.allclose(off, torch.ones_like(off), atol=1e-4))
