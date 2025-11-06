# bats_module.py
# BATS: Budget-Adaptive Tiny-Mamba for Streaming Visual Anomaly Detection
# Wires backbone + budget controller + spectral regularizer + simple anomaly head
# Author: Changyu Li

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any, List

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules
from bats_backbone import TinyMambaBackbone
from budget_controller import BudgetController
from spectral_consistency import SpectralConsistency

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

class CudaTimer:
    """End-to-end latency timer. Uses CUDA events if available, else wall-clock."""
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev = torch.cuda.Event(enable_timing=True)
        self.t0 = 0.0

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_ev.record()
        else:
            self.t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self.use_cuda:
            self.end_ev.record()
            self.end_ev.synchronize()
            return float(self.start_ev.elapsed_time(self.end_ev))  # ms
        else:
            return (time.perf_counter() - self.t0) * 1000.0

def reduce_map_to_score(anom_map: torch.Tensor, topk: float = 0.02) -> torch.Tensor:
    """
    Convert per-pixel anomaly map (B,1,H,W) to per-image scalar score (B,).
    Uses mean of top-k% pixels, a robust alternative to pure max.
    """
    B, _, H, W = anom_map.shape
    k = max(1, int(H * W * topk))
    flat = anom_map.view(B, -1)
    topk_vals, _ = torch.topk(flat, k=k, dim=1)
    return topk_vals.mean(dim=1)

# --------------------------------------------------------------------------------------
# Streaming state (feature buffer + optional mask smoothing)
# --------------------------------------------------------------------------------------

class StreamingState:
    """
    Holds temporal buffers for streaming evaluation:
      - feat_seq: queue of last T feature maps (e.g., C5)
      - mask_ema: optional EMA of saliency masks (for keep-mode smoothing)
    """
    def __init__(self, T: int = 32, ema_alpha: float = 0.8):
        self.T = int(T)
        self.ema_alpha = float(ema_alpha)
        self.feat_seq: List[torch.Tensor] = []
        self.mask_ema: Optional[torch.Tensor] = None  # (B,N) boolean prob (float)

    def reset(self):
        self.feat_seq.clear()
        self.mask_ema = None

    def push_feature(self, feat_c5: torch.Tensor):
        # store a detached copy to keep memory predictable
        self.feat_seq.append(feat_c5.detach())
        if len(self.feat_seq) > self.T:
            self.feat_seq.pop(0)

    def get_seq_tensor(self) -> Optional[torch.Tensor]:
        """
        Stack to (B,C,T,H,W) if we have enough frames, else None.
        Assumes constant batch size across pushes.
        """
        if len(self.feat_seq) == 0:
            return None
        # shape each: (B,C,H,W)
        B, C, H, W = self.feat_seq[0].shape
        T = len(self.feat_seq)
        x = torch.stack(self.feat_seq, dim=2)  # (B,C,T,H,W)
        return x

    def smooth_mask(self, mask_now: torch.Tensor) -> torch.Tensor:
        """
        Simple EMA smoothing for keep-mode boolean mask probabilities.
        mask_now: (B,N) in {0,1} -> cast to float, EMA, then threshold at 0.5
        """
        m = mask_now.float()
        if self.mask_ema is None:
            self.mask_ema = m
        else:
            self.mask_ema = self.ema_alpha * self.mask_ema + (1.0 - self.ema_alpha) * m
        return (self.mask_ema >= 0.5).to(mask_now.dtype)

# --------------------------------------------------------------------------------------
# Anomaly head (lightweight)
# --------------------------------------------------------------------------------------

class AnomalyHead(nn.Module):
    """
    Simple pixel-wise anomaly head on top of pyramid features.
    By default uses only P3 (highest resolution pyramid) with a shallow head.
    You can extend to fuse P3..P5 if needed.
    """
    def __init__(self, in_ch: int = 128, mid: int = 128, out_ch: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, 1, 1, 0, bias=True),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args: feats dict with P3 key at least.
        Returns: anomaly map (B,1,H3,W3) aligned to P3.
        """
        p3 = feats["P3"]
        return self.head(p3)

# --------------------------------------------------------------------------------------
# BATS core model
# --------------------------------------------------------------------------------------

class BATSModel(nn.Module):
    """
    Core model for BATS:
      - TinyMambaBackbone -> (P3,P4,P5)
      - BudgetController (optional during inference/training)
      - AnomalyHead producing per-pixel map
      - Optional SpectralConsistency regularizer over a temporal buffer (C5 by default)

    Training targets:
      - If 'gt_mask' (B,1,H,W) is provided, compute BCE loss on anomaly map (resized).
      - Optionally adds spectral regularization if streaming_state accumulates T frames.

    Forward returns:
      dict {
         'anom_map': (B,1,h,w),
         'score': (B,),
         'latency_ms': float,
         'meta': {budget info, etc.}
      }
    """
    def __init__(
        self,
        backbone_cfg: str = "tiny",
        anomaly_in_ch: int = 128,   # matches FPN out_ch in backbone ('tiny' -> 128)
        use_budget: bool = True,
        level_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        default_budget_q: float = 1.0,
        default_budget_mode: str = "keep",  # 'keep' or 'merge'
        use_spectral_reg: bool = False,
        spectral_kwargs: Optional[Dict[str, Any]] = None,
        streaming_T: int = 32,
        streaming_mask_ema: float = 0.8,
        topk_ratio: float = 0.02
    ):
        super().__init__()
        # Backbone
        self.backbone = TinyMambaBackbone.from_config(backbone_cfg)
        # Budget controller
        self.use_budget = use_budget
        self.default_budget_q = float(default_budget_q)
        self.default_budget_mode = default_budget_mode
        if use_budget:
            self.controller = BudgetController(
                in_channels_p3p4p5=(anomaly_in_ch, anomaly_in_ch, anomaly_in_ch),
                level_weights=level_weights,
            )
        else:
            self.controller = None
        # Anomaly head
        self.head = AnomalyHead(in_ch=anomaly_in_ch, mid=anomaly_in_ch, out_ch=1)
        self.topk_ratio = float(topk_ratio)

        # Spectral regularizer (training-time)
        self.use_spectral_reg = use_spectral_reg
        if use_spectral_reg:
            kwargs = dict(
                win_sizes=[16, 32],
                hop_ratio=0.25,
                channels_first=True,
                pool="gap",
                fps=30.0,
                bands_hz=[],
                coherence_metric="l2",
                lambda_coh=1.0,
                lambda_band=0.0,
                lambda_smooth=0.0,
                reduce="mean"
            )
            if spectral_kwargs:
                kwargs.update(spectral_kwargs)
            self.spec_reg = SpectralConsistency(**kwargs)
        else:
            self.spec_reg = None

        # Streaming state (created lazily per stream)
        self.streaming_template = dict(T=streaming_T, ema_alpha=streaming_mask_ema)

        # Losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    # ----------------------------
    # helpers
    # ----------------------------

    def new_streaming_state(self) -> StreamingState:
        """Create a fresh StreamingState for online evaluation."""
        return StreamingState(T=self.streaming_template["T"], ema_alpha=self.streaming_template["ema_alpha"])

    def _apply_budget(
        self,
        feats: Dict[str, torch.Tensor],
        budget: Optional[float],
        mode: Optional[str],
        streaming_state: Optional[StreamingState]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply budget controller. If streaming_state is provided and mode='keep',
        smooth the boolean mask to avoid frame-to-frame flicker.
        """
        if not self.use_budget or self.controller is None:
            return feats, {}

        q = self.default_budget_q if budget is None else float(budget)
        mode = (mode or self.default_budget_mode).lower()
        out, meta = self.controller(feats, budget=q, mode=mode, return_meta=True)

        # Optional mask smoothing for keep-mode
        if mode == "keep" and streaming_state is not None:
            for lvl in ("P3", "P4", "P5"):
                if f"{lvl}_mask" in meta:
                    smoothed = streaming_state.smooth_mask(meta[f"{lvl}_mask"])
                    # Re-apply smoothed mask (convert to float)
                    x = feats[lvl]
                    B, C, H, W = x.shape
                    N = H * W
                    xm = x.view(B, C, N)
                    xm = xm * smoothed.float().unsqueeze(1)
                    out[lvl] = xm.view(B, C, H, W)
        return out, meta

    # ----------------------------
    # forward
    # ----------------------------

    def forward(
        self,
        images: torch.Tensor,                         # (B,3,H,W)
        budget: Optional[float] = None,
        budget_mode: Optional[str] = None,            # 'keep' | 'merge'
        streaming_state: Optional[StreamingState] = None,
        return_feats: bool = False,
        measure_latency: bool = False
    ) -> Dict[str, Any]:
        timer = CudaTimer() if measure_latency else None
        if timer:
            timer.start()

        # 1) Backbone -> pyramid
        feats = self.backbone(images)                 # has C2..C5 and P3..P5
        pyramid = {k: feats[k] for k in ("P3", "P4", "P5")}

        # 2) (Optional) budget adaptation
        pyramid_adapted, meta = self._apply_budget(pyramid, budget, budget_mode, streaming_state)

        # 3) Anomaly map
        anom_map_logits = self.head(pyramid_adapted)  # (B,1,H3,W3)
        anom_map = torch.sigmoid(anom_map_logits)     # (B,1,H3,W3)
        score = reduce_map_to_score(anom_map, topk=self.topk_ratio)  # (B,)

        # 4) Update streaming buffer with C5
        if streaming_state is not None:
            streaming_state.push_feature(feats["C5"])

        latency_ms = timer.stop_ms() if timer else None

        out = {
            "anom_map": anom_map,
            "logits": anom_map_logits,
            "score": score,
            "meta": meta
        }
        if latency_ms is not None:
            out["latency_ms"] = latency_ms
        if return_feats:
            out["feats"] = feats
        return out

    # ----------------------------
    # loss computation
    # ----------------------------

    def compute_loss(
        self,
        out: Dict[str, Any],
        gt_mask: Optional[torch.Tensor] = None,       # (B,1,H_gt,W_gt) in {0,1}
        streaming_state: Optional[StreamingState] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss:
          - BCE on pixel map (if gt_mask provided)
          - Optional spectral consistency on temporal buffer (C5)
        Returns a dict with total loss and components.
        """
        total = out["logits"].new_zeros(())
        losses = {}

        # Pixel-wise supervision if masks are available
        if gt_mask is not None:
            # Resize GT to prediction size
            _, _, Hp, Wp = out["logits"].shape
            gt = F.interpolate(gt_mask.float(), size=(Hp, Wp), mode="nearest")
            loss_bce = self.bce(out["logits"], gt)
            losses["loss_bce"] = loss_bce
            total = total + loss_bce

        # Spectral regularization (requires temporal buffer)
        if self.use_spectral_reg and (streaming_state is not None) and (self.spec_reg is not None):
            seq = streaming_state.get_seq_tensor()  # (B,C,T,H,W) or None
            if seq is not None and seq.shape[2] >= max(self.spec_reg.win_sizes):
                spec_out = self.spec_reg(seq)       # dict with 'loss'
                losses["loss_spec"] = spec_out["loss"]
                total = total + spec_out["loss"]

        losses["loss"] = total
        return losses

# --------------------------------------------------------------------------------------
# Optional: Lightning wrapper (for quick experiments / CVPR appendix)
# --------------------------------------------------------------------------------------

try:
    import pytorch_lightning as pl

    class BATSSystem(pl.LightningModule):
        """
        LightningModule wrapper to run BATS with a standard training loop.
        This is optional; the core model is framework-agnostic.
        """
        def __init__(self, model: BATSModel, lr: float = 1e-3, weight_decay: float = 0.0):
            super().__init__()
            self.save_hyperparameters(ignore=["model"])
            self.model = model
            self.lr = lr
            self.weight_decay = weight_decay
            self.streaming_state = self.model.new_streaming_state() if self.model.use_spectral_reg else None

        def configure_optimizers(self):
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            return optim

        def training_step(self, batch, batch_idx):
            """
            Expects batch like:
              images: (B,3,H,W)
              gt_mask (optional): (B,1,H,W) in {0,1}
            """
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            gt = batch.get("mask") if isinstance(batch, dict) else None

            out = self.model(images, budget=None, budget_mode=None,
                             streaming_state=self.streaming_state, return_feats=False, measure_latency=False)
            losses = self.model.compute_loss(out, gt_mask=gt, streaming_state=self.streaming_state)
            self.log_dict({k: v for k, v in losses.items()}, on_step=True, on_epoch=True, prog_bar=True)
            # Example metric: mean image score for monitoring (not a training objective)
            self.log("train_score_mean", out["score"].mean(), on_step=True, on_epoch=True, prog_bar=True)
            return losses["loss"]

        def validation_step(self, batch, batch_idx):
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            gt = batch.get("mask") if isinstance(batch, dict) else None

            out = self.model(images, budget=self.model.default_budget_q, budget_mode=self.model.default_budget_mode,
                             streaming_state=self.streaming_state, return_feats=False, measure_latency=True)
            losses = self.model.compute_loss(out, gt_mask=gt, streaming_state=self.streaming_state)
            # Basic logs
            log_vals = {f"val_{k}": v for k, v in losses.items()}
            self.log_dict(log_vals, on_epoch=True, prog_bar=True)
            self.log("val_latency_ms", torch.tensor(out.get("latency_ms", 0.0), device=self.device),
                     on_epoch=True, prog_bar=True)
            self.log("val_score_mean", out["score"].mean(), on_epoch=True, prog_bar=True)

        def on_validation_epoch_start(self):
            if self.streaming_state is not None:
                self.streaming_state.reset()

        def on_train_epoch_start(self):
            if self.streaming_state is not None:
                self.streaming_state.reset()

except ImportError:
    # Lightning is optional; training can be done with plain PyTorch loops.
    BATSSystem = None

# --------------------------------------------------------------------------------------
# Minimal smoke test
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BATSModel(
        backbone_cfg="tiny",
        anomaly_in_ch=128,
        use_budget=True,
        default_budget_q=0.3,
        default_budget_mode="keep",
        use_spectral_reg=True,
        spectral_kwargs=dict(win_sizes=[16,32], hop_ratio=0.25, lambda_coh=1.0, lambda_band=0.0, lambda_smooth=0.0),
        streaming_T=32,
        streaming_mask_ema=0.8,
        topk_ratio=0.02
    ).to(device)

    # Dummy batch
    x = torch.randn(2, 3, 256, 256, device=device)
    gt = (torch.rand(2, 1, 256, 256, device=device) > 0.97).float()  # sparse blobs

    # Streaming state
    stream = model.new_streaming_state()

    # Inference forward
    out = model(x, budget=0.3, budget_mode="keep", streaming_state=stream, return_feats=False, measure_latency=True)
    print("anom_map:", list(out["anom_map"].shape), "score:", out["score"], "latency_ms:", out.get("latency_ms"))

    # Compute loss (as if training)
    losses = model.compute_loss(out, gt_mask=gt, streaming_state=stream)
    print({k: float(v) for k, v in losses.items()})
