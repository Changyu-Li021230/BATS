# bats_backbone.py
# Tiny-Mamba vision backbone with 2D selective-scan + FPN
# Author: Changyu Li 

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Utilities
# ----------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm: normalize across C for each (H,W)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def conv_3x3(in_ch, out_ch, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=bias)


def conv_1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)


# ----------------------------
# Patch / Downsample modules
# ----------------------------

class ConvStem(nn.Module):
    """
    Simple conv stem to get stride-4 features (C2 resolution).
    In: (B,3,H,W) -> (B,C,H/4,W/4)
    """
    def __init__(self, out_channels: int = 64):
        super().__init__()
        c = out_channels
        self.conv1 = nn.Sequential(
            conv_3x3(3, c//2, stride=2, bias=False),
            LayerNorm2d(c//2),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            conv_3x3(c//2, c, stride=2, bias=False),
            LayerNorm2d(c),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Downsample(nn.Module):
    """
    Stage downsample by 2 using stride-2 conv (export friendly).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Sequential(
            LayerNorm2d(in_ch),
            conv_3x3(in_ch, out_ch, stride=2, bias=False)
        )

    def forward(self, x):
        return self.op(x)


# ----------------------------
# Tiny-Mamba 2D selective scan block
# ----------------------------

class SelectiveScan2D(nn.Module):
    """
    A lightweight 2D selective-scan approximating Mamba's SSM behavior for images.
    Idea:
      - Get per-channel gates alpha/beta/gamma via 1x1 convs (content-conditioned).
      - Perform bidirectional recurrent scans along rows, then along cols.
      - Combine local conv path (depthwise) with scan path (SSM-like).
    This keeps it pure-PyTorch and export-friendly, while capturing long-range deps.
    """
    def __init__(self, dim: int, dw_kernel: int = 5):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.local = nn.Sequential(
            conv_3x3(dim, dim, stride=1, groups=dim, bias=False),  # depthwise local mixing
            conv_1x1(dim, dim, bias=True)
        )
        # parameterize gates from content
        self.param = conv_1x1(dim, dim * 3, bias=True)  # alpha, beta, gamma
        self.out_proj = conv_1x1(dim, dim, bias=True)
        self.act = nn.SiLU()

    @staticmethod
    def _scan_1d(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, dim_len: int, along_width: bool):
        """
        x: (B,C,H,W)
        alpha/beta/gamma: (B,C,H,W) gates in [0,1] range after sigmoid
        We scan along W or H. Do forward and backward, then sum.
        Recurrence per step t: s_t = alpha_t * s_{t-1} + beta_t * x_t; y_t = gamma_t * s_t
        """
        B,C,H,W = x.shape
        if along_width:
            # forward
            s = torch.zeros((B,C,H,1), dtype=x.dtype, device=x.device)
            y_f = []
            for t in range(W):
                at = alpha[:,:,:,t:t+1]
                bt = beta [:,:,:,t:t+1]
                gt = gamma[:,:,:,t:t+1]
                xt = x   [:,:,:,t:t+1]
                s = at * s + bt * xt
                y_f.append(gt * s)
            y_f = torch.cat(y_f, dim=3)
            # backward
            s = torch.zeros((B,C,H,1), dtype=x.dtype, device=x.device)
            y_b = []
            for t in reversed(range(W)):
                at = alpha[:,:,:,t:t+1]
                bt = beta [:,:,:,t:t+1]
                gt = gamma[:,:,:,t:t+1]
                xt = x   [:,:,:,t:t+1]
                s = at * s + bt * xt
                y_b.append(gt * s)
            y_b.reverse()
            y_b = torch.cat(y_b, dim=3)
            return y_f + y_b
        else:
            # along height
            s = torch.zeros((B,C,1,W), dtype=x.dtype, device=x.device)
            y_f = []
            for t in range(H):
                at = alpha[:,:,t:t+1,:]
                bt = beta [:,:,t:t+1,:]
                gt = gamma[:,:,t:t+1,:]
                xt = x   [:,:,t:t+1,:]
                s = at * s + bt * xt
                y_f.append(gt * s)
            y_f = torch.cat(y_f, dim=2)
            s = torch.zeros((B,C,1,W), dtype=x.dtype, device=x.device)
            y_b = []
            for t in reversed(range(H)):
                at = alpha[:,:,t:t+1,:]
                bt = beta [:,:,t:t+1,:]
                gt = gamma[:,:,t:t+1,:]
                xt = x   [:,:,t:t+1,:]
                s = at * s + bt * xt
                y_b.append(gt * s)
            y_b.reverse()
            y_b = torch.cat(y_b, dim=2)
            return y_f + y_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalize
        z = self.norm(x)

        # Local (depthwise conv) branch
        local_feat = self.local(z)

        # Gate parameters (content-conditioned)
        gates = self.param(z)
        alpha, beta, gamma = torch.chunk(gates, 3, dim=1)
        alpha = torch.sigmoid(alpha)
        beta  = torch.sigmoid(beta)
        gamma = torch.sigmoid(gamma)

        # 2D selective scans
        y_w = self._scan_1d(z, alpha, beta, gamma, dim_len=z.shape[-1], along_width=True)
        y_h = self._scan_1d(z, alpha, beta, gamma, dim_len=z.shape[-2], along_width=False)

        y = local_feat + 0.5 * (y_w + y_h)
        y = self.act(y)
        y = self.out_proj(y)
        # Residual
        return x + y


class MambaBlock(nn.Module):
    """FFN sandwich + SelectiveScan2D (PreNorm)."""
    def __init__(self, dim: int, ffn_ratio: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        self.ssm = SelectiveScan2D(dim)
        hidden = int(dim * ffn_ratio)
        self.norm = LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            conv_1x1(dim, hidden),
            nn.GELU(),
            conv_1x1(hidden, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.ssm(x))
        # PreNorm FFN
        y = self.ffn(self.norm(x))
        x = x + self.drop_path(y)
        return x


class DropPath(nn.Module):
    """Stochastic depth (per-sample). Export-safe when p=0 at inference."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ----------------------------
# Stage & Backbone
# ----------------------------

class Stage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, drop_path_rates: List[float]):
        super().__init__()
        self.down = Downsample(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        blocks = []
        for i in range(depth):
            blocks.append(MambaBlock(out_ch, ffn_ratio=2.0, drop_path=drop_path_rates[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class TinyMambaBackbone(nn.Module):
    """
    Outputs:
      dict with C2,C3,C4,C5 and P3,P4,P5 (FPN).
    Default channels are small (edge-friendly). Adjust in from_config().
    """
    def __init__(
        self,
        dims: Tuple[int,int,int,int] = (64, 128, 256, 512),
        depths: Tuple[int,int,int,int] = (2, 2, 4, 2),
        drop_path_rate: float = 0.1,
        fpn_out: int = 128
    ):
        super().__init__()
        assert len(dims) == 4 and len(depths) == 4
        self.stem = ConvStem(out_channels=dims[0])

        # stochastic depth schedules
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        i0, i1, i2, i3 = 0, depths[0], depths[0]+depths[1], depths[0]+depths[1]+depths[2]

        self.stage2 = Stage(dims[0], dims[0], depths[0], dpr[i0:i1])     # stride 4
        self.stage3 = Stage(dims[0], dims[1], depths[1], dpr[i1:i2])     # stride 8
        self.stage4 = Stage(dims[1], dims[2], depths[2], dpr[i2:i3])     # stride 16
        self.stage5 = Stage(dims[2], dims[3], depths[3], dpr[i3:])       # stride 32

        # FPN
        self.lateral3 = conv_1x1(dims[1], fpn_out)
        self.lateral4 = conv_1x1(dims[2], fpn_out)
        self.lateral5 = conv_1x1(dims[3], fpn_out)

        self.out3 = conv_3x3(fpn_out, fpn_out, stride=1)
        self.out4 = conv_3x3(fpn_out, fpn_out, stride=1)
        self.out5 = conv_3x3(fpn_out, fpn_out, stride=1)

        # Optional normalization heads (useful for anomaly heatmaps)
        self.norm_c2 = LayerNorm2d(dims[0])
        self.norm_c3 = LayerNorm2d(dims[1])
        self.norm_c4 = LayerNorm2d(dims[2])
        self.norm_c5 = LayerNorm2d(dims[3])

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # upsample x to y's size then add (nearest, export friendly)
        if x.shape[-2:] != y.shape[-2:]:
            x = F.interpolate(x, size=y.shape[-2:], mode='nearest')
        return x + y

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
          x: (B,3,H,W) with H,W multiple of 32 preferred.
        Returns:
          dict { 'C2','C3','C4','C5','P3','P4','P5' }
        """
        # Stem -> C2 resolution
        c2 = self.stem(x)
        c2 = self.stage2(c2)
        c2n = self.norm_c2(c2)

        c3 = self.stage3(c2)
        c3n = self.norm_c3(c3)

        c4 = self.stage4(c3)
        c4n = self.norm_c4(c4)

        c5 = self.stage5(c4)
        c5n = self.norm_c5(c5)

        # FPN
        p5 = self.lateral5(c5n)
        p4 = self._upsample_add(p5, self.lateral4(c4n))
        p3 = self._upsample_add(p4, self.lateral3(c3n))

        p3 = self.out3(p3)
        p4 = self.out4(p4)
        p5 = self.out5(p5)

        return {
            "C2": c2n, "C3": c3n, "C4": c4n, "C5": c5n,
            "P3": p3,  "P4": p4,  "P5": p5
        }

    @staticmethod
    def from_config(name: str = "tiny") -> "TinyMambaBackbone":
        """
        Predefined small/medium configs. You can extend here to match ablation variants.
        """
        if name == "tiny":
            return TinyMambaBackbone(
                dims=(64, 128, 256, 512),
                depths=(2, 2, 4, 2),
                drop_path_rate=0.1,
                fpn_out=128
            )
        elif name == "nano":
            return TinyMambaBackbone(
                dims=(48, 96, 192, 384),
                depths=(2, 2, 3, 2),
                drop_path_rate=0.05,
                fpn_out=96
            )
        elif name == "small":
            return TinyMambaBackbone(
                dims=(80, 160, 320, 512),
                depths=(3, 3, 6, 3),
                drop_path_rate=0.2,
                fpn_out=160
            )
        else:
            raise ValueError(f"Unknown config: {name}")


# ----------------------------
# Minimal sanity check
# ----------------------------

if __name__ == "__main__":
    model = TinyMambaBackbone.from_config("tiny")
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = model(x)
    for k, v in feats.items():
        print(k, list(v.shape))
