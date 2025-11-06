# spectral_consistency.py
# Spectral-consistency regularization for streaming anomaly detection 
# Author: Changyu Li

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------------------
# Core utilities
# --------------------------------------------------------------------------------------

def _hann_window(L: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a Hann window of length L."""
    # identical to librosa.hann but tiny and torch-native
    n = torch.arange(L, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n / (L - 1 + 1e-12)))

def _welch_rfft(x: Tensor, win_size: int, hop: int) -> Tuple[Tensor, int]:
    """
    Welch-style periodogram using 1D rFFT over time.
    Args:
      x: (B, T, D) time series (already spatially pooled/flattened), float
      win_size: window length
      hop: hop length (stride between windows)
    Returns:
      S: (B, F, D) magnitude spectrum averaged over windows, F = win_size//2+1
      F: number of frequency bins
    """
    B, T, D = x.shape
    if T < win_size:
        # zero-pad to minimum length
        pad = win_size - T
        x = F.pad(x, (0, 0, 0, pad))  # pad along T
        T = win_size

    # Unfold into overlapping frames of length win_size with hop
    # shape -> (B, n_frames, win_size, D)
    n_frames = 1 + (T - win_size) // max(1, hop)
    if n_frames <= 0:
        n_frames = 1
        hop = 1
    idx = torch.arange(win_size, device=x.device)
    # frame start positions
    starts = torch.arange(0, (n_frames - 1) * hop + 1, hop, device=x.device)
    # gather frames: we use advanced indexing for clarity
    # x: (B,T,D) -> (B, n_frames, win_size, D)
    frames = x[:, starts[:, None] + idx[None, :], :]  # (B, n_frames, win_size, D)

    # Apply Hann window
    w = _hann_window(win_size, x.device, x.dtype).view(1, 1, win_size, 1)
    frames = frames * w

    # rFFT across the window dimension, keep real frequencies
    # Output shape: (B, n_frames, F, D) complex
    spec = torch.fft.rfft(frames, n=win_size, dim=2)  # complex64/128 depending on dtype
    mag = spec.abs()  # (B, n_frames, F, D)

    # Average across frames (Welch)
    S = mag.mean(dim=1)  # (B, F, D)
    return S, S.shape[1]

def _l2_normalize(x: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    """L2-normalize along a dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def _band_mask(fps: float, nfft: int, bands_hz: List[Tuple[float, float]], device, dtype) -> Tensor:
    """
    Create a stack of frequency band masks (triangular or binary). Here we use binary masks for simplicity.
    Args:
      fps: sampling rate in Hz (frames per second)
      nfft: number of rFFT bins = win_size//2 + 1
      bands_hz: list of (low_hz, high_hz)
    Returns:
      M: (BANDS, F) binary mask (0/1)
    """
    # Frequency value for each bin k is f[k] = k * fps / win_size, but with rFFT we only have 0..nfft-1.
    # We approximate using bin frequencies evenly spaced up to Nyquist.
    # Nyquist = fps/2, step ~ (fps/2)/(nfft-1)
    f = torch.linspace(0.0, fps / 2.0, nfft, device=device, dtype=dtype)  # (F,)
    masks = []
    for lo, hi in bands_hz:
        m = (f >= max(0.0, lo)) & (f < max(lo, hi))
        masks.append(m.to(dtype))
    if len(masks) == 0:
        return torch.empty(0, nfft, device=device, dtype=dtype)
    return torch.stack(masks, dim=0)  # (BAND, F)

# --------------------------------------------------------------------------------------
# Public loss module
# --------------------------------------------------------------------------------------

class SpectralConsistency(nn.Module):
    """
    Spectral Consistency Regularizer (CVPR-friendly, export-safe).

    Goal:
      Encourage the temporal frequency response of features to be coherent across adjacent time scales
      (e.g., 16 vs 32 frame windows). Optionally, constrain energy within physics-inspired bands.

    Typical use:
      1) Collect a short temporal buffer of features (e.g., C5 logits or pre-head maps).
      2) Call this module to obtain a scalar loss you can add to your main objective.

    Input shapes:
      - seq can be (B, T, C, H, W) or (B, C, T, H, W). Set channels_first accordingly.
      - We first spatially pool to (B, T, D) (D=C or C*P if you choose to flatten spatially).

    Loss terms:
      - L_coh: cross-scale spectral coherence. L2 distance (or 1 - cosine sim) between
               unit-normalized spectra from consecutive window sizes.
      - L_band: optional band energy alignment (between scales, or toward a target profile).
      - L_smooth: optional time-domain smoothness (Huber on first or second differences).

    Args:
      win_sizes: list of window sizes (e.g., [16, 32]) used for rFFT/Welch.
      hop_ratio: hop length = max(1, int(win_size * hop_ratio)).
      channels_first: whether seq is (B, C, T, H, W). If False, expects (B, T, C, H, W).
      pool: 'gap' (global avg pool over H,W) or 'flatten' (avg over patches then concat).
      fps: frames per second (needed only if you set bands_hz).
      bands_hz: optional list of (low_hz, high_hz) for band energy constraints.
      coherence_metric: 'l2' or 'cosine' comparing normalized spectra.
      lambda_coh / lambda_band / lambda_smooth: weights for each term.
      diff_order: 1 or 2 for time-domain smoothness order (Δx or Δ²x).
      reduce: 'mean' or 'sum'.

    Notes:
      - All ops are torch-native (rFFT, unfold-style Welch), no third-party deps.
      - Export-friendly; avoid dynamic control flow in the forward pass.
    """
    def __init__(
        self,
        win_sizes: List[int] = [16, 32],
        hop_ratio: float = 0.25,
        channels_first: bool = True,
        pool: str = "gap",
        fps: float = 30.0,
        bands_hz: Optional[List[Tuple[float, float]]] = None,
        coherence_metric: str = "l2",  # 'l2' or 'cosine'
        lambda_coh: float = 1.0,
        lambda_band: float = 0.2,
        lambda_smooth: float = 0.0,
        diff_order: int = 1,
        reduce: str = "mean"
    ):
        super().__init__()
        assert pool in ("gap", "flatten")
        assert coherence_metric in ("l2", "cosine")
        assert diff_order in (1, 2)
        assert len(win_sizes) >= 2, "Provide at least two window sizes for cross-scale coherence."

        self.win_sizes = sorted([int(max(4, w)) for w in win_sizes])
        self.hop_ratio = hop_ratio
        self.channels_first = channels_first
        self.pool = pool
        self.fps = float(fps)
        self.bands_hz = bands_hz or []
        self.coherence_metric = coherence_metric
        self.lambda_coh = float(lambda_coh)
        self.lambda_band = float(lambda_band)
        self.lambda_smooth = float(lambda_smooth)
        self.diff_order = diff_order
        self.reduce = reduce

    # ----------------------------
    # helpers
    # ----------------------------

    def _spatial_pool(self, seq: Tensor) -> Tensor:
        """
        Spatially pool features.
        Input:
          - (B, T, C, H, W) or (B, C, T, H, W)
        Output:
          - (B, T, D) where D=C for GAP, or D=C if we avg patches then concat.
            (We keep D=C for both modes to avoid head re-wiring; 'flatten' here means
             average each patch location over time first, then concat across C — but to keep
             export and simplicity we just GAP; switch below if you truly want concat.)
        """
        if self.channels_first:
            # (B, C, T, H, W) -> (B, T, C)
            x = seq
            x = x.mean(dim=-1).mean(dim=-1)  # GAP over H, then W -> (B, C, T)
            x = x.transpose(1, 2).contiguous()  # (B, T, C)
        else:
            # (B, T, C, H, W) -> (B, T, C)
            x = seq.mean(dim=-1).mean(dim=-1)  # GAP over H,W
        return x  # (B, T, C)

    def _normalize_time(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        """Zero-mean, unit-variance along time for stability: (B,T,D)."""
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + eps
        return (x - mu) / sd

    def _coherence(self, S_a: Tensor, S_b: Tensor) -> Tensor:
        """
        Compare two spectra (B, F, D) after L2-norm along F.
        Return per-batch scalar loss (B,).
        """
        # Normalize per (F) so shapes are compatible
        A = _l2_normalize(S_a, dim=1)
        B_ = _l2_normalize(S_b, dim=1)

        if self.coherence_metric == "l2":
            # mean squared error across (F,D)
            diff = (A - B_) ** 2
            loss = diff.mean(dim=(1, 2))
        else:  # cosine distance over F, averaged across D
            # cosine sim along F per channel, then 1 - sim
            num = (A * B_).sum(dim=1)  # (B, D)
            den = (A.norm(dim=1) * B_.norm(dim=1) + 1e-8)  # (B, D), already ~1 after normalize
            cos = num / den
            loss = (1.0 - cos).mean(dim=1)  # (B,)
        return loss  # (B,)

    def _band_energy(self, S: Tensor, bands_mask: Tensor) -> Tensor:
        """
        Compute band-wise energy by summing |S| over frequency bins.
        Args:
          S: (B, F, D) magnitude spectrum
          bands_mask: (BAND, F) in {0,1}
        Returns:
          E: (B, BAND, D) normalized per-sample across BAND (L1)
        """
        if bands_mask.numel() == 0:
            return S.new_zeros(S.shape[0], 0, S.shape[2])
        # (B,F,D) x (BAND,F) -> (B,BAND,D)
        M = bands_mask[None, :, :, None]  # (1,BAND,F,1)
        energy = (S[:, None, :, :] * M).sum(dim=2)  # (B,BAND,D)
        # L1-normalize across bands to compare shapes
        energy = energy / (energy.sum(dim=1, keepdim=True) + 1e-8)
        return energy

    def _temporal_smooth(self, x: Tensor) -> Tensor:
        """
        Time-domain smoothness on (B,T,D). Huber loss on first/second differences.
        Returns a per-batch scalar (B,).
        """
        if self.diff_order == 1:
            dx = x[:, 1:, :] - x[:, :-1, :]
        else:
            dx = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        # Huber (a.k.a. smooth L1)
        hub = F.smooth_l1_loss(dx, torch.zeros_like(dx), reduction='none')
        return hub.mean(dim=(1, 2))  # (B,)

    # ----------------------------
    # forward
    # ----------------------------

    def forward(self, seq: Tensor) -> Dict[str, Tensor]:
        """
        Compute spectral consistency losses.
        Args:
          seq: (B, C, T, H, W) if channels_first=True, else (B, T, C, H, W)
        Returns:
          dict: {
            'loss': total scalar,
            'loss_coh': L_coh,
            'loss_band': L_band,
            'loss_smooth': L_smooth
          }
        """
        # 1) Spatial pooling -> (B, T, D)
        x = self._spatial_pool(seq)                # (B,T,D)
        x = self._normalize_time(x)                # (B,T,D)
        B, T, D = x.shape

        # 2) Compute spectra for all window sizes (Welch rFFT)
        spectra: List[Tensor] = []
        nffts: List[int] = []
        for w in self.win_sizes:
            hop = max(1, int(w * self.hop_ratio))
            S, Fbins = _welch_rfft(x, win_size=int(w), hop=hop)  # (B,F,D)
            spectra.append(S)
            nffts.append(Fbins)

        # 3) Cross-scale coherence between consecutive windows
        loss_coh_terms = []
        for i in range(len(spectra) - 1):
            Sa, Sb = spectra[i], spectra[i + 1]      # (B,Fa,D), (B,Fb,D)
            # Align frequency resolution by linear interpolation over F
            if Sa.shape[1] != Sb.shape[1]:
                # Interpolate along F dimension
                targetF = max(Sa.shape[1], Sb.shape[1])
                Sa = F.interpolate(Sa.permute(0, 2, 1), size=targetF, mode="linear", align_corners=False).permute(0, 2, 1)
                Sb = F.interpolate(Sb.permute(0, 2, 1), size=targetF, mode="linear", align_corners=False).permute(0, 2, 1)
            loss_i = self._coherence(Sa, Sb)  # (B,)
            loss_coh_terms.append(loss_i)
        loss_coh = torch.stack(loss_coh_terms, dim=0).mean(dim=0) if len(loss_coh_terms) else x.new_zeros(B)

        # 4) Optional band energy alignment (between first and last window size)
        if len(self.bands_hz) > 0:
            # Build band masks using the *largest* F resolution for stability
            S_ref = spectra[-1]
            bands_mask = _band_mask(self.fps, S_ref.shape[1], self.bands_hz, device=x.device, dtype=x.dtype)  # (BAND,F)
            Eb = self._band_energy(S_ref, bands_mask)  # (B,BAND,D)
            # Compare to smaller scales' band profiles (or simply to the next scale)
            band_terms = []
            for S in spectra[:-1]:
                if S.shape[1] != S_ref.shape[1]:
                    S = F.interpolate(S.permute(0, 2, 1), size=S_ref.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
                E = self._band_energy(S, bands_mask)  # (B,BAND,D)
                # L2 distance across bands (averaged over D)
                band_terms.append(((E - Eb) ** 2).mean(dim=(1, 2)))  # (B,)
            loss_band = torch.stack(band_terms, dim=0).mean(dim=0) if band_terms else x.new_zeros(B)
        else:
            loss_band = x.new_zeros(B)

        # 5) Optional time-domain smoothness
        loss_smooth = self._temporal_smooth(x) if self.lambda_smooth > 0.0 else x.new_zeros(B)

        # 6) Reduce and aggregate
        if self.reduce == "mean":
            Lc = loss_coh.mean()
            Lb = loss_band.mean()
            Ls = loss_smooth.mean()
        else:
            Lc = loss_coh.sum()
            Lb = loss_band.sum()
            Ls = loss_smooth.sum()

        total = self.lambda_coh * Lc + self.lambda_band * Lb + self.lambda_smooth * Ls

        return {
            "loss": total,
            "loss_coh": Lc.detach(),
            "loss_band": Lb.detach(),
            "loss_smooth": Ls.detach()
        }


# --------------------------------------------------------------------------------------
# Quick sanity check
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Fake sequence with a mild temporal rhythm at ~3 Hz on 30 FPS
    B, C, T, H, W = 2, 64, 48, 8, 8
    fps = 30.0
    t = torch.arange(T).float()
    s = (torch.sin(2 * math.pi * 3.0 * t / fps)[None, None, :, None, None])  # (1,1,T,1,1)
    seq = 0.5 * torch.randn(B, C, T, H, W) + 0.3 * s.expand(B, C, T, H, W)

    reg = SpectralConsistency(
        win_sizes=[16, 32],
        hop_ratio=0.25,
        channels_first=True,
        pool="gap",
        fps=fps,
        bands_hz=[(0.5, 2.0), (2.0, 6.0), (6.0, 12.0)],
        coherence_metric="l2",
        lambda_coh=1.0,
        lambda_band=0.2,
        lambda_smooth=0.0,
        reduce="mean"
    )
    out = reg(seq)
    print({k: float(v) for k, v in out.items()})
