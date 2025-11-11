# BATS: Budget-Adaptive Tiny-Mamba for Streaming Visual Anomaly Detection

> **Anytime inference under tight latency on edge devices.**  
> BATS couples a Tiny-Mamba vision backbone with a budget controller (token keep/merge) and a spectral-consistency regularizer to achieve early, stable anomaly detection in streaming industrial imagery.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-CVPR--style--code--release-brightgreen)

**Repository**: https://github.com/Changyu-Li021230/BATS  
**Paper target**: ICML (Streaming AD / Efficient Vision)

---

## ✨ Highlights

- **Anytime inference**: continuous budget–accuracy trade-off via token **keep** (sparse masking) or **merge** (resolution reduction).  
- **Streaming stability**: **SpectralConsistency** aligns temporal frequency responses across scales → fewer flickers/fragmentation & more stable MTTD.  
- **Edge-friendly**: tiny backbone + FPN, export-safe ops (dw-conv/1×1/avg-pool), easy to port to Jetson/TensorRT.

---

## 👥 Authors & Maintainers

- **Changyu Li** (GitHub: `@Changyu-Li021230`) — lead author & maintainer  
- **Jiaxin Chen** — co-author  
- **Fei Luo** — co-author / advisor

> Contact: please open a GitHub issue for bug reports or feature requests.

---

## 🧠 Method at a Glance

- **Backbone — `TinyMambaBackbone`**  
  Lightweight 2D selective-scan block (Mamba-style SSM) with local depthwise mixing and content-gated row/column scans; emits `{C2..C5, P3..P5}` via FPN.

- **Budget Controller — `BudgetController` (on `{P3,P4,P5}`)**  
  - **keep**: Top-k saliency masking (keeps spatial size; decoder-friendly).  
  - **merge**: non-overlapping average pooling to approximate a target token count (ToMe-like, export-safe).

- **Spectral Consistency — `SpectralConsistency`**  
  Welch rFFT over temporal buffers; enforces cross-scale spectral coherence, optional physics-band priors; reduces jitter & alarm fragmentation.

- **Head**  
  Shallow pixel head on **P3** → per-pixel anomaly maps; per-image score via top-k pooling.

---

## 🗂️ Repository Structure

```
bats_backbone.py          # Tiny-Mamba + FPN backbone
budget_controller.py      # Budget-adaptive keep/merge for P3/P4/P5
spectral_consistency.py   # Spectral regularizer for streaming stability
bats_module.py            # End-to-end model + loss + (optional) LightningModule
streaming_protocol.py     # Online protocol, decisions, metrics, stream builder
eval_stream.py            # CVPR-ready evaluation across a budget grid
```

---

## 🛠️ Installation

```bash
# Python 3.10+ recommended
conda create -n bats python=3.10 -y
conda activate bats

# Install PyTorch matching your CUDA/CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core deps
pip install numpy pillow

# Optional (Lightning)
pip install pytorch-lightning

# Optional (plotting, out of repo)
pip install matplotlib pandas
```

**Tested environments**
- Python 3.10/3.11, PyTorch 2.2–2.4, CUDA 12.x  
- Linux x86_64 + NVIDIA GPU; Jetson Orin via export (see Export section)

---

## ⚡ Quick Start

### 1) Smoke test
```bash
python bats_module.py
```
You should see a dummy forward/loss and one-pass latency readout.

### 2) Streaming evaluation (folder-based toy example)
Prepare two folders of RGB images:
- `normal_dir` — normal frames
- `anomaly_dir` — anomaly frames

```bash
python eval_stream.py \
  --normal_dir /path/to/normal_images \
  --anomaly_dir /path/to/anomaly_images \
  --use_budget --mode keep \
  --budgets 1.0 0.75 0.5 0.3 0.2 0.1 \
  --repeat 5 --len_prefix 120 --len_anomaly 40 --len_suffix 80 \
  --amp --warmup_iters 30 \
  --out_dir outputs/eval_stream_keep
```

Outputs (CSV/JSON) → `outputs/eval_stream_keep/`:
- `frames_q*.csv` — per-frame scores, alarms, latency  
- `streams_q*.csv` — per-stream MTTD/FAR/Fragmentation/latency quantiles/AUROC  
- `summary.csv` & `summary.json` — aggregated results per budget

---

## 📊 Streaming Evaluation

`streaming_protocol.py` provides:
- **Stream builder**: `StreamingConcat` to create **Normal → Anomaly → Normal** sequences  
- **State**: `StreamingState` buffers features and optional mask-EMA  
- **Decision**: `OnlineDecision` with EMA + hysteresis + cooldown  
- **Metrics**: **MTTD**, **FAR**, **Fragmentation**, **e2e latency** (p50/p90/p99)

Switch budget mode:
```bash
# keep-mode (sparse masking, same resolution)
python eval_stream.py --mode keep --use_budget ...

# merge-mode (avg-pool merge, fewer tokens)
python eval_stream.py --mode merge --use_budget ...
```

---

## 🧪 Training Hooks (Optional)

Minimal BCE supervision is provided for pixel masks; spectral regularizer can be enabled for stability.

```python
from bats_module import BATSModel
model = BATSModel(
    backbone_cfg="tiny",
    use_budget=True,
    default_budget_q=1.0,
    default_budget_mode="keep",
    use_spectral_reg=True,
    spectral_kwargs=dict(win_sizes=[16,32], hop_ratio=0.25,
                         lambda_coh=1.0, lambda_band=0.0, lambda_smooth=0.0)
).to(device)

stream_state = model.new_streaming_state()
out = model(images.to(device), budget=None, budget_mode=None,
            streaming_state=stream_state, measure_latency=False)
losses = model.compute_loss(out, gt_mask=gt_mask.to(device), streaming_state=stream_state)
(losses["loss"]).backward()
```

Lightning users: use `BATSSystem` in `bats_module.py`.

---

## 🎛️ Budgets & Modes

- **Budget** `q ∈ (0,1]` — target token fraction per pyramid level (allocated by level weights).  
- **keep**: Top-k saliency mask, preserve spatial size; best drop-in for conv decoders.  
- **merge**: non-overlapping avg-pool to ~target tokens; reduces spatial size & compute.

For **anytime inference**, randomize budgets during training and evaluate across a grid.

---

## 📐 CLI Arguments (eval_stream.py)

| Argument | Type | Default | Description |
|---|---:|:---:|---|
| `--normal_dir` | str | – | Folder with normal frames (RGB). |
| `--anomaly_dir` | str | – | Folder with anomaly frames (RGB). |
| `--len_prefix` | int | 120 | Normal prefix length. |
| `--len_anomaly` | int | 40 | Anomaly segment length. |
| `--len_suffix` | int | 80 | Normal suffix length. |
| `--repeat` | int | 5 | How many streams to build. |
| `--budgets` | float+ | `1.0 0.75 0.5 0.3 0.2 0.1` | Budget list (q). |
| `--mode` | str | `keep` | `keep` or `merge`. |
| `--use_budget` | flag | off | Enable budget controller. |
| `--ema_alpha` | float | 0.9 | Score EMA smoothing factor. |
| `--on_thresh` | float | 0.6 | Hysteresis ON threshold. |
| `--off_thresh` | float | 0.5 | Hysteresis OFF threshold. |
| `--cooldown` | int | 10 | Cooldown frames after alarm. |
| `--noise_std` | float | 0.0 | Additive Gaussian noise (0..1 img scale). |
| `--brightness_drift` | float | 0.0 | Per-frame brightness drift. |
| `--device` | str | `cuda` | Device for inference. |
| `--seed` | int | 2026 | RNG seed. |
| `--out_dir` | str | `outputs/eval_stream` | Output directory. |
| `--backbone_cfg` | str | `tiny` | `nano` \| `tiny` \| `small`. |
| `--use_spec` | flag | off | Enable spectral regularizer (train-time only). |
| `--amp` | flag | off | Use AMP during inference for latency runs. |
| `--warmup_iters` | int | 0 | Warmup iterations before timing. |

---

## 🔁 Reproducibility & Latency Reporting

- Fix seeds: `--seed` sets Python/NumPy/PyTorch RNGs.  
- Stable latency: use `--amp` & `--warmup_iters` (e.g., 30–50); report **e2e latency** with p50/p90/p99.  
- Keep `spectral_consistency` **off** at export/inference; it’s a **training-time** regularizer.

---

## 📦 Export (ONNX / TensorRT) — Sketch

BATS uses export-safe ops (conv, depthwise, avg-pool, 1×1, simple elementwise). Typical steps:

```python
# 1) Switch model to eval; disable spectral regularizer
model = BATSModel(backbone_cfg="tiny", use_budget=True, use_spectral_reg=False).eval().to("cuda")

# 2) Dummy input
dummy = torch.randn(1, 3, 256, 256, device="cuda")

# 3) ONNX export (keep-mode default; set q=1.0 for a fixed graph if preferred)
torch.onnx.export(
    model, (dummy,), "bats.onnx",
    input_names=["images"], output_names=["anom_map","logits","score"],
    opset_version=17, dynamic_axes={"images": {0:"B", 2:"H", 3:"W"}}
)
```

> TensorRT: import `bats.onnx`, enable FP16, set workspace; consider fusing the head & post-proc where convenient.  
> Jetson: prefer `--mode keep` + fixed `q` for a stable graph (or calibrate INT8 after reviewing quality impact).

---

## 🧾 Dataset Notes

Benchmarks to integrate:
- **MVTec-AD** (15 classes, pixel masks)  
- **VisA** (12 classes, 10k+ images)

This repo’s script uses folder-based streams for simplicity. To get AUPRO/pixel PR:
1) wrap your dataset to yield `(image, mask)` for `StreamingConcat`, or  
2) adapt loaders from Anomalib and pass through `SimpleDatasetWrapper`.

---

## 📈 Results Placeholder (fill after experiments)

- **Main table** (MVTec-AD / VisA): AUROC/AUPRO, Params, FLOPs, e2e Latency (p50/p90/p99)  
- **Pareto frontier**: Latency(p99) or Params vs. AUPRO (multi-budget q)  
- **Budget curves**: q → AUPRO, q → Latency  
- **Streaming stability**: Detection-delay CDF, Fragmentation vs. baselines  
- **Robustness**: AUPRO/MTTD retention under noise/blur/drift

---

## 🗺️ Roadmap / TODO

- [ ] Multi-level fusion head (P3–P5) + lightweight decoder  
- [ ] Native AUPRO / pixel-PR in `eval_stream.py`  
- [ ] Jetson/TensorRT export recipe & benchmarks  
- [ ] Per-class adapters (MVTec/VisA loaders)  
- [ ] More robustness ops (blur, geometric drift, SNR curves)

---

## ❓ FAQ

**Q1: Should I use `keep` or `merge`?**  
`keep` preserves spatial size—plug-and-play with conv heads, often better early detection under small budgets. `merge` reduces resolution for more consistent speedups; validate on your data.

**Q2: Where do I apply the spectral regularizer?**  
Train time only, based on a temporal buffer of C5 features; disable for export/inference.

**Q3: Any gotchas for latency measurement?**  
Warm up (30–50 iters), fix seeds, match AMP setting to baselines, report p50/p90/p99, and clarify full e2e path (decode→preproc→model→postproc→decision).

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Li2026BATS,
  title     = {BATS: Budget-Adaptive Tiny-Mamba for Streaming Visual Anomaly Detection},
  author    = {Changyu Li },
  year      = {2026}
}
```

*(Update once the paper is on arXiv/OpenReview.)*

---

## 📜 License & Acknowledgements

- **License:** Code is released under **CC BY 4.0** (feel free to switch to MIT/BSD-3-Clause as needed).  
- **Acknowledgements:** Inspired by state-space models (Mamba/VMamba), token selection/merging (DynamicViT/ToMe), and open anomaly detection tooling (e.g., Anomalib). Thanks to the open-source community.
