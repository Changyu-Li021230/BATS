# eval_stream.py
# Run streaming evaluation for BATS across budgets and export metrics.
# Author: Changyu Li


from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn

# Local modules
from bats_module import BATSModel
from streaming_protocol import (
    StreamConfig, StreamingConcat, StreamRunner, OnlineMetrics,
    SimpleDatasetWrapper
)

# --------------------------------------------------------------------------------------
# Minimal image dataset helpers (optional)
# --------------------------------------------------------------------------------------

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class ImageFolderNoLabel:
    """
    Minimal (image-only) dataset from a directory of images.
    Files under `root` will be loaded and converted to float32 [0,1] tensors (C,H,W).
    Masks are not provided (None). If you need masks, plug your own dataset class.
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, root: str, size: Optional[int] = None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")
        self.files = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in self.IMG_EXTS])
        if size is not None:
            self.files = self.files[:size]
        if PIL_AVAILABLE is False:
            raise RuntimeError("PIL not available. Install pillow or provide your own dataset.")
        if len(self.files) == 0:
            raise RuntimeError(f"No image files under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3)
        x = x.permute(2, 0, 1).contiguous()                  # (3,H,W)
        return {"image": x, "mask": None}


# --------------------------------------------------------------------------------------
# Metrics/Utilities
# --------------------------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def env_report() -> Dict[str, Any]:
    rep = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "deterministic_algos": torch.are_deterministic_algorithms_enabled()
    }
    if torch.cuda.is_available():
        rep["gpu_name"] = torch.cuda.get_device_name(0)
    return rep

def quantiles(x: List[float], qs=(0.5, 0.9, 0.99)) -> Dict[str, float]:
    arr = np.array(x, dtype=np.float64)
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(arr, q)) if arr.size > 0 else float("nan")
    out["mean"] = float(arr.mean()) if arr.size > 0 else float("nan")
    return out

def roc_auc_score_simple(y_true: List[int], y_score: List[float]) -> float:
    """
    Minimal ROC-AUC without sklearn.
    Uses the Mann–Whitney U relation: AUC = (rank sum of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    Ties are handled by average ranking.
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # argsort scores to obtain ranks; handle ties by average ranks
    order = np.argsort(s, kind="mergesort")  # stable
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # average ranks for ties
    uniq, idx_start = np.unique(s[order], return_index=True)
    for i, start in enumerate(idx_start):
        end = idx_start[i + 1] if i + 1 < len(idx_start) else len(s)
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg

    sum_pos = ranks[y == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_csv(rows: List[Dict[str, Any]], path: Path, header: Optional[List[str]] = None):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is None and len(rows) > 0:
        header = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def build_streams_from_folders(
    normal_dir: Optional[str],
    anomaly_dir: Optional[str],
    len_prefix: int,
    len_anomaly: int,
    len_suffix: int,
    repeat: int
) -> List[StreamingConcat]:
    """
    Build multiple Normal→Anomaly→Normal streams from folder datasets.
    If dirs are None, raises; otherwise constructs 'repeat' streams.
    """
    if normal_dir is None or anomaly_dir is None:
        raise ValueError("Please provide both --normal_dir and --anomaly_dir, or integrate your own dataset loader.")
    normal_ds = ImageFolderNoLabel(normal_dir)
    anomaly_ds = ImageFolderNoLabel(anomaly_dir)
    streams = []
    for _ in range(repeat):
        streams.append(
            StreamingConcat(
                normal_ds=normal_ds,
                anomaly_ds=anomaly_ds,
                len_prefix=len_prefix,
                len_anomaly=len_anomaly,
                len_suffix=len_suffix,
                shuffle_each_segment=True
            )
        )
    return streams

def evaluate_budget_grid(
    model: BATSModel,
    streams: List[StreamingConcat],
    budgets: List[float],
    mode: str,
    cfg_template: StreamConfig,
    out_dir: Path
):
    """
    For each budget q, evaluate all streams and export per-frame CSV + summary CSV + JSON.
    """
    summary_rows = []
    env = env_report()
    all_results = {"env": env, "results": {}}

    # Apply budget mode globally (either keep or merge)
    model.default_budget_mode = mode

    for q in budgets:
        model.default_budget_q = float(q)
        per_stream_metrics = []
        per_frame_rows: List[Dict[str, Any]] = []

        for s_idx, stream in enumerate(streams):
            # Fresh config & runner each stream (ensures clean state)
            cfg = StreamConfig(
                ema_alpha=cfg_template.ema_alpha,
                on_thresh=cfg_template.on_thresh,
                off_thresh=cfg_template.off_thresh,
                cooldown=cfg_template.cooldown,
                max_frames=cfg_template.max_frames,
                noise_std=cfg_template.noise_std,
                brightness_drift=cfg_template.brightness_drift,
                device=cfg_template.device
            )
            runner = StreamRunner(model, cfg)
            out = runner.run(stream)

            # Per-frame bookkeeping for CSV
            rec = out["records"]
            T = len(rec["score"])
            for t in range(T):
                per_frame_rows.append({
                    "budget_q": q,
                    "stream_id": s_idx,
                    "frame_idx": rec["frame_idx"][t],
                    "label": rec["label"][t],
                    "score": f"{rec['score'][t]:.6f}",
                    "score_ema": f"{rec['score_ema'][t]:.6f}",
                    "alarm": rec["alarm"][t],
                    "rising": rec["rising"][t],
                    "latency_ms": f"{rec['latency_ms'][t]:.4f}"
                })

            # Online metrics
            m: OnlineMetrics = out["metrics"]
            lat = quantiles(rec["latency_ms"])  # p50/p90/p99/mean
            # AUROC over frames using raw score (not smoothed)
            auroc = roc_auc_score_simple(rec["label"], rec["score"])

            per_stream_metrics.append({
                "budget_q": q,
                "stream_id": s_idx,
                "MTTD_frames": (float(m.mttd) if m.mttd is not None else np.nan),
                "FAR_per_1k": float(m.far_per_1k),
                "Fragmentation_per_1k": float(m.fragmentation),
                "Detections": int(m.detections),
                "Total_frames": int(m.total_frames),
                "Latency_p50_ms": lat["p50"],
                "Latency_p90_ms": lat["p90"],
                "Latency_p99_ms": lat["p99"],
                "Latency_mean_ms": lat["mean"],
                "Frame_AUROC": float(auroc)
            })

        # Aggregate across streams
        agg = {}
        for k in ["MTTD_frames", "FAR_per_1k", "Fragmentation_per_1k", "Detections", "Total_frames",
                  "Latency_p50_ms", "Latency_p90_ms", "Latency_p99_ms", "Latency_mean_ms", "Frame_AUROC"]:
            vals = [row[k] for row in per_stream_metrics if not (isinstance(row[k], float) and math.isnan(row[k]))]
            agg[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

        # Write per-budget CSVs
        write_csv(per_frame_rows, out_dir / f"frames_q{q:.2f}.csv")
        write_csv(per_stream_metrics, out_dir / f"streams_q{q:.2f}.csv")

        # Add to summary
        sum_row = {"budget_q": q, **agg}
        summary_rows.append(sum_row)
        all_results["results"][f"q{q:.2f}"] = {"aggregate": agg}

    # Final summary
    write_csv(summary_rows, out_dir / "summary.csv")
    write_json(all_results, out_dir / "summary.json")

    print(f"[eval_stream] Wrote results to: {out_dir.resolve()}")
    print("Summary (first few rows):")
    for r in summary_rows[:min(5, len(summary_rows))]:
        print(r)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BATS streaming evaluation (CVPR-ready).")
    # Data
    p.add_argument("--normal_dir", type=str, default=None,
                   help="Folder of normal frames (images). If None, provide your own dataset logic.")
    p.add_argument("--anomaly_dir", type=str, default=None,
                   help="Folder of anomaly frames (images). If None, provide your own dataset logic.")
    p.add_argument("--len_prefix", type=int, default=120, help="Normal prefix length.")
    p.add_argument("--len_anomaly", type=int, default=40, help="Anomaly segment length.")
    p.add_argument("--len_suffix", type=int, default=80, help="Normal suffix length.")
    p.add_argument("--repeat", type=int, default=5, help="How many streams to build.")
    # Budget & mode
    p.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.3, 0.2, 0.1],
                   help="List of budget q values.")
    p.add_argument("--mode", type=str, default="keep", choices=["keep", "merge"], help="Budget mode.")
    # Online decision
    p.add_argument("--ema_alpha", type=float, default=0.9)
    p.add_argument("--on_thresh", type=float, default=0.6)
    p.add_argument("--off_thresh", type=float, default=0.5)
    p.add_argument("--cooldown", type=int, default=10)
    # Perturbations
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--brightness_drift", type=float, default=0.0)
    # Runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out_dir", type=str, default="outputs/eval_stream")
    # Model knobs
    p.add_argument("--backbone_cfg", type=str, default="tiny", choices=["nano", "tiny", "small"])
    p.add_argument("--use_budget", action="store_true", help="Enable budget controller (default off here).")
    p.add_argument("--use_spec", action="store_true", help="Enable spectral regularizer (training-time only).")
    # Latency stability
    p.add_argument("--amp", action="store_true", help="Enable AMP during inference for latency runs.")
    p.add_argument("--warmup_iters", type=int, default=0, help="Warmup iterations before timing.")
    return p.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[env]", json.dumps(env_report(), indent=2))

    # Build model
    model = BATSModel(
        backbone_cfg=args.backbone_cfg,
        use_budget=args.use_budget,
        default_budget_q=1.0,
        default_budget_mode=args.mode,
        use_spectral_reg=args.use_spec,
        spectral_kwargs=dict(win_sizes=[16, 32], hop_ratio=0.25, lambda_coh=1.0, lambda_band=0.0, lambda_smooth=0.0),
        streaming_T=32,
        streaming_mask_ema=0.8,
        topk_ratio=0.02
    )
    # AMP & warmup knobs (from earlier patch)
    model.use_amp = bool(args.amp)
    model.warmup_iters = int(args.warmup_iters)

    # Build streams (you can replace this with your dataset-specific logic)
    streams = build_streams_from_folders(
        normal_dir=args.normal_dir,
        anomaly_dir=args.anomaly_dir,
        len_prefix=args.len_prefix,
        len_anomaly=args.len_anomaly,
        len_suffix=args.len_suffix,
        repeat=args.repeat
    )

    # Stream config template
    cfg = StreamConfig(
        ema_alpha=args.ema_alpha,
        on_thresh=args.on_thresh,
        off_thresh=args.off_thresh,
        cooldown=args.cooldown,
        max_frames=None,
        noise_std=args.noise_std,
        brightness_drift=args.brightness_drift,
        device=args.device
    )

    # Evaluate grid
    evaluate_budget_grid(
        model=model,
        streams=streams,
        budgets=args.budgets,
        mode=args.mode,
        cfg_template=cfg,
        out_dir=out_dir
    )

if __name__ == "__main__":
    main()
