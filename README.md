# Multi-Model Chest X-Ray Inference on a $249 Jetson Orin Nano Super

**How many disease-detection models can one $249 edge box run at once, and how
fast?** This repo measures it end to end — from naive PyTorch to quantized
TensorRT — on a single NVIDIA Jetson Orin Nano Super (8 GB, 25 W).

> Systems / capability study, **not** a clinical product. Pretrained
> [torchxrayvision](https://github.com/mlmed/torchxrayvision) models are used as-is —
> no retraining. The value is the *systems* result: concurrency, throughput, power,
> accuracy trade-offs, cost.

## The headline

The same box, same models, optimized step by step:

| | Throughput | |
|---|---:|---|
| Naive PyTorch (1 model, 1 image) | 20 img/s | 1× |
| Best PyTorch (concurrent + batched) | 180 img/s | 9× |
| TensorRT FP16 (batched) | 509 img/s | 25× |
| **TensorRT INT8 (batched)** | **1035 img/s** | **52×** |

…and every trade-off along the way is measured and reported honestly — including the
negative results (naive TTA, the INT8 accuracy cost, the concurrency memory wall).

**Reproducibility.** Every experiment was repeated and reported with error bars —
±1 standard error over 3 runs for throughput, ±1 bootstrap SE (1000 resamples) for
AUROC. Accuracy reproduces **exactly** (AUROC is bit-identical run to run) and
throughput variance is **under 1%** across the board. Sustained load is fine too: a
re-run 10-minute endurance test held 508 img/s (−0.2%, 69 °C, no throttling). One
operational note — pack cooldowns between back-to-back heavy runs, or accumulated
heat transiently throttles the GPU clock.

## Experiments

Each folder is one self-contained experiment with its own README, code, and result.

| # | Experiment | Headline result |
|---|---|---|
| [XP1](experiments/xp01_baselines/) | Single-model baselines | batching → 160 img/s, linear; the efficiency king |
| [XP2](experiments/xp02_concurrency/) | Concurrency + memory wall | sublinear, saturates ~6 models; N=8 exceeds 8 GB |
| [XP3](experiments/xp03_concurrent_batched/) | Concurrent + batched | 180 img/s @ 97% GPU — the device ceiling |
| [XP4](experiments/xp04_hetero_panel/) | Mixed-architecture panel | ResNet-50 + 3 DenseNets, 75 img/s; heavy model ≠ bottleneck |
| [XP5](experiments/xp05_cuda_streams/) | CUDA streams | fixes memory (12 models, 680 MB), not throughput (serialized) |
| [XP6](experiments/xp06_tensorrt_fp16/) | TensorRT FP16 | 509 img/s (25×), accuracy preserved to 0.86 pp |
| [XP7](experiments/xp07_int8/) | INT8 quantization | 1035 img/s (2×) but −0.054 AUROC — screening vs diagnosis |
| [XP8](experiments/xp08_robustness/) | TTA / ensemble | ensemble +0.022 AUROC ~free; naive TTA doesn't help |
| [XP9](experiments/xp09_power_modes/) | Power-envelope sweep | MAXN for peak; 25 W for best efficiency |
| [XP10](experiments/xp10_endurance/) | Thermal endurance | −0.2% over 20 min, 71 °C — no throttling |
| [XP11](experiments/xp11_serving/) | Serving layer + load | dynamic batching; SLA-safe capacity ~482 req/s (raw 510) |
| [XP12](experiments/xp12_energy_governor/) | Energy governor | adaptive power scaling; −3.4 % energy vs MAXN — but power mode is a weak lever |
| [demos](demos/) | Live demos | PyTorch (75 img/s) · TensorRT (398 img/s) · browser replay |

## The clinic story

- One **$249 box** runs a **multi-disease chest X-ray panel locally** — no cloud,
  patient images never leave the building.
- **No subscription:** one-time hardware pays back a comparable cloud GPU in
  ~4 months (`results/figures/cost_comparison.png`).
- Different clinicians ask different questions **at the same time** — different models,
  which can't be batched, so concurrency is the point.

## Repository layout

```
lib/            shared code: models, utils, power_logger, trt_runner, chest_labels
experiments/    xp01 … xp10, each: README + scripts (+ shared engines/data referenced)
demos/          PyTorch demo, TensorRT demo, self-contained browser replay
results/        raw/ (per-run JSON corpus) · figures/ (generated) · *.json summaries
analysis/       make_figures.py — regenerates every figure from results/
PLAN.md         the original build spec · requirements.txt — Jetson-specific setup
```

Scripts run directly (`python <script>.py`); each adds `lib/` to its path. Rebuild all
figures with `python analysis/make_figures.py`. Environment setup (Jetson PyTorch +
cuDSS/cuBLAS fixes, TensorRT, medmnist) is documented in `requirements.txt`.

## Honesty guardrails

- Pretrained models' **published** accuracy only — no clinical validity claims.
- The $249 is a **hook**, not a hard constraint (it also runs on bigger boxes).
- "True parallel" on one GPU = interleaved kernels; gains are **sublinear** — the
  curve is the finding. Negative results are reported, not hidden.
