# budget_controller.py
# Budget-adaptive token keep / merge for pyramid features 
# Author: Changyu Li

from __future__ import annotations
from typing import Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ----------------------------
# Utilities
# ----------------------------

def _spatial_tokens(x: Tensor) -> Tuple[Tensor, Tuple[int,int]]:
    """Convert (B,C,H,W) to (B, H*W, C) and return (H, W)."""
    B, C, H, W = x.shape
    t = x.view(B, C, H*W).transpose(1, 2).contiguous()
    return t, (H, W)

def _from_tokens(tokens: Tensor, hw: Tuple[int,int], C: int) -> Tensor:
    """Convert (B, N, C) back to (B, C, H, W)."""
    B, N, _ = tokens.shape
    H, W = hw
    out = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
    return out

def _topk_mask(scores: Tensor, k: int) -> Tensor:
    """
    scores: (B, N). Return boolean mask (B, N) where Top-k positions are True.
    """
    B, N = scores.shape
    k = max(1, min(k, N))
    topk = torch.topk(scores, k=k, dim=1).indices  # (B, k)
    mask = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, topk, True)
    return mask

def _apply_mask_keep(x: Tensor, mask: Tensor) -> Tensor:
    """
    Keep mode: zero-out non-Top-k positions while preserving spatial resolution.
    x: (B,C,H,W), mask: (B,N) with N = H*W
    Return: (B,C,H,W)
    """
    B, C, H, W = x.shape
    N = H * W
    xm = x.view(B, C, N)
    mask_f = mask.float()  # (B, N)
    xm = xm * mask_f.unsqueeze(1)  # broadcast along channel
    return xm.view(B, C, H, W)

def _avg_pool_reduce(x: Tensor, target_tokens: int) -> Tuple[Tensor, float]:
    """
    Merge mode (ToMe-like approximation): use non-overlapping avg pool to approach
    a target token count by choosing an integer stride.
    Input:  x (B,C,H,W)
    Output: y (B,C,H',W') and scale = H'/H (same for W).
    """
    B, C, H, W = x.shape
    # Approximate target grid side length ~ sqrt(target_tokens)
    grid = int(math.sqrt(max(1, target_tokens)))
    # Integer strides so that H' * W' ≈ target_tokens (export-friendly)
    stride_h = max(1, H // max(1, grid))
    stride_w = max(1, W // max(1, grid))
    y = F.avg_pool2d(x, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w), ceil_mode=False)
    scale_h = y.shape[-2] / H
    return y, scale_h

# ----------------------------
# Saliency head
# ----------------------------

class SaliencyHead(nn.Module):
    """
    Lightweight per-token saliency scorer:
      score_map = Conv1x1(GELU(Conv1x1(x))) -> (B,1,H,W) -> flattened to (B,N).
    Intuitively approximates channel energy with learnable projections.
    """
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        s = self.proj(x)        # (B,1,H,W)
        B, _, H, W = s.shape
        return s.view(B, H * W) # (B,N)

# ----------------------------
# Budget Controller
# ----------------------------

class BudgetController(nn.Module):
    """
    Budget-adaptive controller for pyramid features:
      - mode='keep': keep Top-k tokens by zero-masking others (preserve spatial size).
      - mode='merge': reduce resolution by non-overlapping avg pooling (ToMe-like).
    Budget q ∈ (0,1]: the fraction of tokens to keep or approximate after reduction.
    Per-level allocation is controlled by 'level_weights'.
    """
    def __init__(
        self,
        in_channels_p3p4p5: Tuple[int,int,int] = (128, 128, 128),
        level_weights: Tuple[float,float,float] = (0.5, 0.3, 0.2),
        min_tokens_per_level: int = 16,
        default_mode: str = "keep"
    ):
        super().__init__()
        assert len(in_channels_p3p4p5) == 3
        assert len(level_weights) == 3 and abs(sum(level_weights) - 1.0) < 1e-6
        self.level_weights = level_weights
        self.min_tokens = min_tokens_per_level
        self.default_mode = default_mode

        c3, c4, c5 = in_channels_p3p4p5
        self.sal_p3 = SaliencyHead(c3)
        self.sal_p4 = SaliencyHead(c4)
        self.sal_p5 = SaliencyHead(c5)

    @torch.no_grad()
    def _count_tokens(self, x: Tensor) -> int:
        """Return H*W for a (B,C,H,W) tensor."""
        H, W = x.shape[-2:]
        return H * W

    def _alloc_per_level(self, feats: Dict[str, Tensor], q: float) -> Dict[str, int]:
        """
        Allocate target token counts across P3/P4/P5 according to level_weights,
        respecting min_tokens and not exceeding each level's tokens.
        """
        total_tokens = sum(self._count_tokens(feats[k]) for k in ("P3", "P4", "P5"))
        target_total = max(1, int(total_tokens * max(1e-4, min(1.0, q))))

        alloc = {}
        for key, w in zip(("P3", "P4", "P5"), self.level_weights):
            n = self._count_tokens(feats[key])
            alloc[key] = max(self.min_tokens, int(target_total * w))
            alloc[key] = min(alloc[key], n)

        # Small correction loop to match the requested total as closely as possible
        diff = target_total - sum(alloc.values())
        keys = ["P3", "P4", "P5"]
        i = 0
        while diff != 0:
            k = keys[i % 3]
            nmax = self._count_tokens(feats[k])
            if diff > 0 and alloc[k] < nmax:
                alloc[k] += 1
                diff -= 1
            elif diff < 0 and alloc[k] > self.min_tokens:
                alloc[k] -= 1
                diff += 1
            i += 1
            if i > 10000:  # safety break
                break
        return alloc

    def _scores_per_level(self, feats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute saliency scores (B,N) for each level."""
        return {
            "P3": self.sal_p3(feats["P3"]),
            "P4": self.sal_p4(feats["P4"]),
            "P5": self.sal_p5(feats["P5"]),
        }

    def forward(
        self,
        feats: Dict[str, Tensor],           # expects keys: P3/P4/P5
        budget: Optional[float] = None,     # q in (0,1]; None => 1.0 (no reduction)
        mode: Optional[str] = None,         # 'keep' | 'merge'
        return_meta: bool = True
    ) -> Dict[str, Tensor] | Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Returns:
          - adapted pyramid dict: {P3, P4, P5}
          - optional meta dict (masks, k per level, scale factors)
        """
        assert "P3" in feats and "P4" in feats and "P5" in feats, "feats must include P3/P4/P5"
        q = float(1.0 if budget is None else max(1e-4, min(1.0, budget)))
        mode = (mode or self.default_mode).lower()
        assert mode in ("keep", "merge")

        out: Dict[str, Tensor] = {}
        meta: Dict[str, Tensor] = {}

        # 1) Target tokens per level
        alloc = self._alloc_per_level(feats, q)

        if mode == "keep":
            # 2) Saliency -> Top-k mask -> zero-out others (preserve spatial size)
            scores = self._scores_per_level(feats)
            for level in ("P3", "P4", "P5"):
                x = feats[level]
                B, C, H, W = x.shape
                N = H * W
                k = max(1, min(alloc[level], N))
                mask = _topk_mask(scores[level], k)          # (B,N)
                y = _apply_mask_keep(x, mask)                # (B,C,H,W)
                out[level] = y
                if return_meta:
                    meta[f"{level}_k"] = torch.tensor([k], device=x.device, dtype=torch.int32)
                    meta[f"{level}_mask"] = mask
                    meta[f"{level}_scale"] = torch.tensor([1.0], device=x.device, dtype=torch.float32)

        else:  # mode == "merge"
            # 2) Non-overlapping avg pool to approximate target tokens
            for level in ("P3", "P4", "P5"):
                x = feats[level]
                y, scale = _avg_pool_reduce(x, target_tokens=alloc[level])
                out[level] = y
                if return_meta:
                    meta[f"{level}_k"] = torch.tensor([y.shape[-2] * y.shape[-1]], device=x.device, dtype=torch.int32)
                    meta[f"{level}_scale"] = torch.tensor([scale], device=x.device, dtype=torch.float32)

        return (out, meta) if return_meta else out


# ----------------------------
# Quick sanity check
# ----------------------------
if __name__ == "__main__":
    B, H, W = 2, 128, 128
    P3 = torch.randn(B, 128, H//8,  W//8)   # 16x16
    P4 = torch.randn(B, 128, H//16, W//16)  # 8x8
    P5 = torch.randn(B, 128, H//32, W//32)  # 4x4
    feats = {"P3": P3, "P4": P4, "P5": P5}

    ctrl = BudgetController(in_channels_p3p4p5=(128,128,128), level_weights=(0.5,0.3,0.2))
    ctrl.eval()

    # KEEP: zero-out non-Top-k while preserving resolution
    (y_keep, meta_k) = ctrl(feats, budget=0.3, mode="keep", return_meta=True)
    for k,v in y_keep.items():
        print("KEEP", k, list(v.shape), "k=", int(meta_k[f"{k}_k"].item()))

    # MERGE: reduce resolution via average pooling
    (y_merge, meta_m) = ctrl(feats, budget=0.3, mode="merge", return_meta=True)
    for k,v in y_merge.items():
        print("MERGE", k, list(v.shape), "k=", int(meta_m[f"{k}_k"].item()))
