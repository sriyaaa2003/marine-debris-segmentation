"""DeepSpectralUNet: a residual attention U-Net whose decoder receives a multi-scale branch built only from the four
spectral indices (FDI, NDVI, NDWI, FAI). Input is 15 channels: 11 Sentinel-2 bands followed by the four indices."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
N_BANDS = 11
N_INDICES = 4


class AttentionGate(nn.Module):
    """Attention gate (Oktay et al. 2018): the decoder signal g decides which skip-connection pixels to pass on."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * self.psi(self.relu(self.W_g(g_up) + self.W_x(x)))


class CBAM(nn.Module):
    """Channel then spatial attention (Woo et al. 2018)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel_att(x).view(x.size(0), x.size(1), 1, 1)
        avg_p = x.mean(dim=1, keepdim=True)
        max_p = x.max(dim=1, keepdim=True)[0]
        return x * self.spatial_att(torch.cat([avg_p, max_p], dim=1))


class ResConvBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection, which keeps gradient flowing to the rare debris pixels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x):
        return self.relu(self.drop(self.conv(x)) + self.skip(x))


class MultiScaleSpectralBranch(nn.Module):
    """The four indices at full resolution and at every encoder scale; one feature map per scale."""

    def __init__(self, n_indices=N_INDICES, features=(64, 128, 256, 512)):
        super().__init__()
        self.scales = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        in_ch = n_indices
        for f in features:
            self.scales.append(nn.Sequential(nn.Conv2d(in_ch, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True)))
            in_ch = f

    def forward(self, x):
        outputs = []
        for scale in self.scales:
            x = scale(x)
            outputs.append(x)
            x = self.pool(x)
        return outputs


class DeepSpectralUNet(nn.Module):
    def __init__(self, in_channels=N_BANDS + N_INDICES, num_classes=NUM_CLASSES, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.enc = nn.ModuleList()
        self.enc_cbam = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(ResConvBlock(ch, f))
            self.enc_cbam.append(CBAM(f))
            ch = f
        self.bottleneck = nn.Sequential(
            ResConvBlock(features[-1], features[-1] * 2),
            ResConvBlock(features[-1] * 2, features[-1] * 2),
            ResConvBlock(features[-1] * 2, features[-1] * 2),
        )
        self.spectral_branch = MultiScaleSpectralBranch(N_INDICES, features)
        self.up_convs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.dec_cbam = nn.ModuleList()
        self.dec = nn.ModuleList()
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.att_gates.append(AttentionGate(F_g=f, F_l=f, F_int=max(f // 2, 1)))
            self.dec_cbam.append(CBAM(f))
            self.dec.append(ResConvBlock(f * 2 + f, f))
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        spectral_maps = self.spectral_branch(x[:, N_BANDS:N_BANDS + N_INDICES])[::-1]
        skips = []
        for conv, cbam in zip(self.enc, self.enc_cbam):
            x = cbam(conv(x))
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i, (up, att, dcbam, dec) in enumerate(zip(self.up_convs, self.att_gates, self.dec_cbam, self.dec)):
            x = up(x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            skip = dcbam(att(g=x, x=skip))
            spec = spectral_maps[i]
            if spec.shape[2:] != x.shape[2:]:
                spec = F.interpolate(spec, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x, spec], dim=1))
        return self.final_conv(x)
