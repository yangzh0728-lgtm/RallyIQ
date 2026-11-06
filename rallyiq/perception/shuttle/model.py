"""Network definitions for the multi-frame heatmap shuttle detector.

* ``HeatmapNet``: U-Net style encoder-decoder. Input is a stack of ``seq_len``
  consecutive RGB frames (optionally preceded by a background median image),
  resized to 512x288; output is one heatmap per input frame.

Layer names match the released pretrained checkpoints so their ``state_dict`` loads
unchanged. Architecture and weights are MIT licensed; see ``LICENSE-THIRD-PARTY``
in this directory.

Imported only by the detector (never at ``rallyiq`` import time) because it needs torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_HEIGHT = 288
INPUT_WIDTH = 512


class _Conv2DBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Double2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = _Conv2DBlock(in_dim, out_dim)
        self.conv_2 = _Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class _Triple2DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = _Conv2DBlock(in_dim, out_dim)
        self.conv_2 = _Conv2DBlock(out_dim, out_dim)
        self.conv_3 = _Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class HeatmapNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.down_block_1 = _Double2DConv(in_dim, 64)
        self.down_block_2 = _Double2DConv(64, 128)
        self.down_block_3 = _Triple2DConv(128, 256)
        self.bottleneck = _Triple2DConv(256, 512)
        self.up_block_1 = _Triple2DConv(768, 256)
        self.up_block_2 = _Double2DConv(384, 128)
        self.up_block_3 = _Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))

    def forward(self, x):
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(F.max_pool2d(x1, 2))
        x3 = self.down_block_3(F.max_pool2d(x2, 2))
        x = self.bottleneck(F.max_pool2d(x3, 2))
        x = self.up_block_1(torch.cat([F.interpolate(x, scale_factor=2.0), x3], dim=1))
        x = self.up_block_2(torch.cat([F.interpolate(x, scale_factor=2.0), x2], dim=1))
        x = self.up_block_3(torch.cat([F.interpolate(x, scale_factor=2.0), x1], dim=1))
        return torch.sigmoid(self.predictor(x))


def heatmap_in_channels(seq_len: int, bg_mode: str) -> int:
    if bg_mode == "concat":
        return (seq_len + 1) * 3
    if bg_mode == "":
        return seq_len * 3
    raise ValueError(f"unsupported background mode {bg_mode!r} (expected '' or 'concat')")


def build_heatmap_net(seq_len: int, bg_mode: str) -> HeatmapNet:
    return HeatmapNet(heatmap_in_channels(seq_len, bg_mode), seq_len)
