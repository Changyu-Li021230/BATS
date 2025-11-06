# streaming_protocol.py
# Online streaming protocol for BATS: build Normal→Anomaly→Normal streams,
# maintain stateful inference, apply online decisions, and compute early-detection metrics.
# Author: Changyu Li

from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any
from dataclasses import dataclass

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from bats_module import BATSModel, StreamingState

Tensor = torch.Tensor


# Configs


@dataclass
class StreamConfig:
    """
    Online decision & stream configuration.
    - ema_alpha: score exponential smoothing factor in [0,1). Larger = smoother.
    - on_thresh/off_thresh: hysteresis thresholds (on >= off). on triggers alarm, off clears it.
    - cooldown: frames to suppress new alarms after one alarm ends (avoid chatter).
    - max_frames: optional truncation for the stream.
    - noise_std: additive Gaussian noise sigma in [0,1] image scale, 0 disables.
    - brightness_drift: per-frame brightness drift (added each frame), 0 disables.
    - device: inference device.
    """
    ema_alpha: float = 0.9
    on_thresh: float = 0.6
    off_thresh: float = 0.5
    cooldown: int = 5
    max_frames: Optional[int] = None

    noise_std: float = 0.0
    brightness_drift: float = 0.0

    device: str = "cuda"



# Perturbations (export-friendly, training-time only)


def apply_perturbations(img: Tensor, frame_idx: int, cfg: StreamConfig) -> Tensor:
    """
    Apply simple robustness perturbations in image space (float, 0..1).
    - Gaussian noise with std = cfg.noise_std
    - Linear brightness drift: add frame_idx * brightness_drift (then clamp)
    """
    x = img
    if cfg.brightness_drift != 0.0:
        x = x + float(frame_idx) * cfg.brightness_drift
    if cfg.noise_std > 0.0:
        x = x + torch.randn_like(x) * cfg.noise_std
    return x.clamp(0.0, 1.0)


# Online decision with EMA + hysteresis + cooldown


class OnlineDecision:
    """
    Maintains smoothed score and produces binary alarms with hysteresis.
    Hysteresis prevents flapping near a single threshold; cooldown prevents rapid re-triggers.
    """
    def __init__(self, ema_alpha: float, on_thresh: float, off_thresh: float, cooldown: int):
        assert on_thresh >= off_thresh, "on_thresh must be >= off_thresh for hysteresis."
        self.ema_alpha = float(ema_alpha)
        self.on = float(on_thresh)
        self.off = float(off_thresh)
        self.cooldown = int(cooldown)

        self.smoothed: Optional[Tensor] = None  # (B,)
        self.state: Optional[Tensor] = None     # (B,) in {0,1}
        self.cool: Optional[Tensor] = None      # (B,) cooldown counters

    def reset(self, batch_size: int, device: torch.device):
        self.smoothed = torch.zeros(batch_size, device=device)
        self.state = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.cool = torch.zeros(batch_size, dtype=torch.long, device=device)

    @torch.no_grad()
    def step(self, score: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Update with new raw score in [0,1], return:
          - state: (B,) current alarm state {0,1}
          - rising: (B,) 1 where a new alarm just triggered at this step
        """
        assert score.dim() == 1, "score should be (B,) after reduce_map_to_score"
        B = score.shape[0]
        device = score.device
        if self.smoothed is None:
            self.reset(B, device)

        # EMA smoothing
        self.smoothed = self.ema_alpha * self.smoothed + (1.0 - self.ema_alpha) * score

        # Apply cooldown decrement
        self.cool = torch.clamp(self.cool - 1, min=0)

        prev = self.state.clone()

        # Hysteresis update per sample
        turn_on = (self.smoothed >= self.on) & (self.cool == 0)
        turn_off = (self.smoothed <= self.off)

        self.state = torch.where(turn_on, torch.ones_like(self.state), self.state)
        self.state = torch.where(turn_off, torch.zeros_like(self.state), self.state)

        # When turning off, set cooldown
        just_off = (prev == 1) & (self.state == 0)
        self.cool = torch.where(just_off, torch.full_like(self.cool, self.cooldown), self.cool)

        rising = (prev == 0) & (self.state == 1)
        return self.state, rising.long()



# Stream builders

def sample_indices(n: int, length: int, replace: bool = False) -> List[int]:
    """Sample 'length' indices from [0..n-1]."""
    assert length > 0
    if replace or length > n:
        return [random.randrange(n) for _ in range(length)]
    else:
        return random.sample(range(n), length)

class SimpleDatasetWrapper:
    """
    Minimal adapter that expects an underlying dataset returning:
      - image: Tensor (C,H,W) in [0,1]
      - mask:  Optional[Tensor] (1,H,W) in {0,1} or None
      If your dataset returns (image, label) instead, set mask=None.
    """
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        item = self.base[idx]
        if isinstance(item, dict):
            img = item.get("image") or item.get("img")
            mask = item.get("mask")  # can be None
        elif isinstance(item, (tuple, list)):
            # try (image, mask) or (image,) convention
            img = item[0]
            mask = item[1] if len(item) > 1 else None
        else:
            img = item
            mask = None
        return {"image": img, "mask": mask}

class StreamingConcat:
    """
    Build a stream by concatenating:
      [Normal-prefix] + [Anomaly] + [Normal-suffix]

    Each segment is sampled from given datasets with specified lengths.
    This class yields dicts with:
      - 'image': (C,H,W) in [0,1]
      - 'mask':  Optional (1,H,W) in {0,1}
      - 'label': 0 for normal, 1 for anomaly
      - 'frame_idx': global frame index in the stream
    """
    def __init__(
        self,
        normal_ds,
        anomaly_ds,
        len_prefix: int,
        len_anomaly: int,
        len_suffix: int,
        shuffle_each_segment: bool = True
    ):
        self.normal = SimpleDatasetWrapper(normal_ds)
        self.anomaly = SimpleDatasetWrapper(anomaly_ds)
        self.Ln1 = int(len_prefix)
        self.La = int(len_anomaly)
        self.Ln2 = int(len_suffix)
        self.shuffle_each_segment = shuffle_each_segment

        self.norm_idx1 = sample_indices(len(self.normal), self.Ln1, replace=len(self.normal) < self.Ln1)
        self.anom_idx = sample_indices(len(self.anomaly), self.La, replace=len(self.anomaly) < self.La)
        self.norm_idx2 = sample_indices(len(self.normal), self.Ln2, replace=len(self.normal) < self.Ln2)

        if shuffle_each_segment:
            random.shuffle(self.norm_idx1)
            random.shuffle(self.anom_idx)
            random.shuffle(self.norm_idx2)

        self.total = self.Ln1 + self.La + self.Ln2

    def __len__(self):
        return self.total

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        frame = 0
        # Normal prefix
        for i in self.norm_idx1:
            item = self.normal[i]
            yield {"image": item["image"], "mask": item["mask"], "label": torch.tensor(0), "frame_idx": frame}
            frame += 1
        # Anomaly
        for i in self.anom_idx:
            item = self.anomaly[i]
            yield {"image": item["image"], "mask": item["mask"], "label": torch.tensor(1), "frame_idx": frame}
            frame += 1
        # Normal suffix
        for i in self.norm_idx2:
            item = self.normal[i]
            yield {"image": item["image"], "mask": item["mask"], "label": torch.tensor(0), "frame_idx": frame}
            frame += 1



# Metrics: MTTD, FAR, Fragmentation


@dataclass
class OnlineMetrics:
    """Container for online metrics."""
    mttd: Optional[float]            # mean time-to-detect (frames), or None if never detected
    far_per_1k: float                # false alarms per 1000 normal frames
    fragmentation: float             # number of on→off transitions per 1000 frames (lower is better)
    detections: int                  # number of alarm onsets
    total_frames: int

def compute_online_metrics(
    labels: List[int],
    rising_events: List[int],
    states: List[int]
) -> OnlineMetrics:
    """
    Compute online metrics using per-frame:
      - labels[t] in {0,1}
      - rising_events[t] in {0,1} (new alarm triggered at t)
      - states[t] in {0,1} (current alarm state)
    MTTD: min (t_rise - t_anom_onset) per anomaly episode, averaged across episodes detected.
    FAR: ratio of rising events that occur during normal frames, normalized per 1k normal frames.
    Fragmentation: number of off→on transitions per 1k frames (overall chattiness).
    """
    T = len(labels)
    # Find anomaly onset frames
    onsets = []
    prev = 0
    for t, y in enumerate(labels):
        if y == 1 and prev == 0:
            onsets.append(t)
        prev = y

    # For each onset, find first rising event at or after onset
    delays = []
    rise_indices = [t for t, r in enumerate(rising_events) if r == 1]
    for onset in onsets:
        det = next((t for t in rise_indices if t >= onset), None)
        if det is not None:
            delays.append(det - onset)

    mttd = (sum(delays) / len(delays)) if len(delays) > 0 else None

    # FAR per 1k normal frames
    normal_frames = sum(1 for y in labels if y == 0)
    false_rises = sum(1 for t, r in enumerate(rising_events) if r == 1 and labels[t] == 0)
    far_per_1k = (false_rises / max(1, normal_frames)) * 1000.0

    # Fragmentation per 1k frames: count off->on transitions (regardless of label)
    on_transitions = 0
    prev_state = 0
    for s in states:
        if prev_state == 0 and s == 1:
            on_transitions += 1
        prev_state = s
    fragmentation = (on_transitions / max(1, T)) * 1000.0

    return OnlineMetrics(
        mttd=mttd,
        far_per_1k=far_per_1k,
        fragmentation=fragmentation,
        detections=on_transitions,
        total_frames=T
    )


# Stream runner (model + state + decision + metrics)


class StreamRunner:
    """
    Run a BATSModel on a streaming iterator frame-by-frame, maintaining:
      - model StreamingState (features across time)
      - OnlineDecision (score smoothing & hysteresis)
      - Optional perturbations for robustness tests
    Produces per-frame records and summary metrics.
    """
    def __init__(self, model: BATSModel, cfg: StreamConfig):
        self.model = model.eval()
        self.cfg = cfg

    @torch.inference_mode()
    def run(self, stream: Iterable[Dict[str, Tensor]]) -> Dict[str, Any]:
        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        state = self.model.new_streaming_state()
        decider = OnlineDecision(
            ema_alpha=self.cfg.ema_alpha,
            on_thresh=self.cfg.on_thresh,
            off_thresh=self.cfg.off_thresh,
            cooldown=self.cfg.cooldown
        )

        records: Dict[str, List[Any]] = {
            "score": [], "score_ema": [], "alarm": [], "rising": [],
            "latency_ms": [], "label": [], "frame_idx": []
        }

        B = None
        t_global = 0
        for item in stream:
            if self.cfg.max_frames is not None and t_global >= self.cfg.max_frames:
                break

            img = item["image"].unsqueeze(0).to(device)  # (1,C,H,W)
            if img.dtype != torch.float32:
                img = img.float()
            # Assume input in 0..1; apply perturbations if requested
            img = apply_perturbations(img, t_global, self.cfg)

            # Forward
            out = self.model(
                images=img,
                budget=self.model.default_budget_q,
                budget_mode=self.model.default_budget_mode,
                streaming_state=state,
                return_feats=False,
                measure_latency=True
            )
            score = out["score"].detach()  # (1,)
            latency = float(out.get("latency_ms", 0.0))

            # Decision update
            alarm_state, rising = decider.step(score)
            if B is None:
                B = int(score.shape[0])
                decider.reset(B, device)

            # Bookkeeping
            records["score"].append(float(score[0].item()))
            records["score_ema"].append(float(decider.smoothed[0].item()))
            records["alarm"].append(int(alarm_state[0].item()))
            records["rising"].append(int(rising[0].item()))
            records["latency_ms"].append(latency)
            records["label"].append(int(item["label"]))
            records["frame_idx"].append(int(item["frame_idx"]))

            t_global += 1

        # Metrics
        metrics = compute_online_metrics(
            labels=records["label"],
            rising_events=records["rising"],
            states=records["alarm"]
        )

        return {"records": records, "metrics": metrics}

# Minimal smoke test (no external dataset required)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy "datasets": constant gray images, anomaly masks only in anomaly part
    class DummyDS:
        def __init__(self, length: int, anomaly: bool = False):
            self.length = length
            self.anomaly = anomaly
        def __len__(self): return self.length
        def __getitem__(self, idx):
            img = torch.full((3, 128, 128), 0.5, dtype=torch.float32)  # 0.5 gray
            mask = torch.zeros(1, 128, 128, dtype=torch.float32)
            if self.anomaly:
                # put a small bright blob as "anomaly"
                y0, x0, r = 64, 64, 10
                Y, X = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
                blob = (((Y - y0)**2 + (X - x0)**2) <= r**2).float()
                img = img + blob.unsqueeze(0) * 0.4
                mask = blob.unsqueeze(0)
            return {"image": img.clamp(0,1), "mask": mask}

    normal_ds = DummyDS(200, anomaly=False)
    anomaly_ds = DummyDS(60, anomaly=True)

    stream = StreamingConcat(
        normal_ds, anomaly_ds,
        len_prefix=120, len_anomaly=40, len_suffix=80,
        shuffle_each_segment=False
    )

    # Light BATS stub if you don't have the full model here:
    try:
        model = BATSModel(
            backbone_cfg="nano",    # smaller for the test
            use_budget=False,       # focus on protocol
            use_spectral_reg=False
        ).to(device)
    except Exception as e:
        print("If BATSModel is unavailable, replace with a stub producing 'score' & 'latency_ms'.")
        raise e

    cfg = StreamConfig(
        ema_alpha=0.9,
        on_thresh=0.6,
        off_thresh=0.5,
        cooldown=10,
        max_frames=None,
        noise_std=0.00,
        brightness_drift=0.0,
        device=device
    )

    runner = StreamRunner(model, cfg)
    out = runner.run(stream)

    print("Online metrics:")
    m = out["metrics"]
    print(f"  MTTD (frames): {m.mttd}")
    print(f"  FAR (/1k normal frames): {m.far_per_1k:.2f}")
    print(f"  Fragmentation (/1k frames): {m.fragmentation:.2f}")
    print(f"  Detections: {m.detections}, Total frames: {m.total_frames}")
