# Multi-Model Chest X-Ray Inference on a $249 Jetson Orin Nano Super

**How many disease-detection models can one $249 edge box run at once, and how
efficiently?** This repo measures it — throughput, power, and energy efficiency
for running chest X-ray pathology models **sequentially vs. batched vs. many at
once** on a single NVIDIA Jetson Orin Nano Super (8 GB, 25 W).

> Systems/capability demo, **not** a clinical product. Pretrained
> [torchxrayvision](https://github.com/mlmed/torchxrayvision) DenseNet-121 models
> are used as-is — no retraining, no accuracy claims. The value is the *systems*
> result: concurrency, throughput, power, cost.

## Headline results

Measured on the board in `MAXN_SUPER` mode, FP16 (autocast), DenseNet-121,
3 repeats per config (low variance), power via `tegrastats` (`VDD_IN`).

| Regime | Config | Throughput | GPU | Power | Efficiency |
|---|---|---:|---:|---:|---:|
| Sequential | 1 model, 1 image | 20.7 img/s | 18% | 8.4 W | 2.5 img/s/W |
| Concurrent only | 6 models × 1 | 100 img/s | 78% | 16.3 W | 6.2 img/s/W |
| Batched only | 1 model × 8 | 167 img/s | 83% | 13.9 W | 11.8 img/s/W |
| **Concurrent + batched** | **4 models × 4** | **180 img/s** | **97%** | 18.2 W | 9.9 img/s/W |

**Five findings** (see `results/figures/`):

1. **Batching scales throughput linearly** with batch size (20 → 167 img/s at
   batch 8) at flat latency — the most *energy-efficient* option for one disease.
   → `batching_baseline.png`
2. **Concurrency is real but sublinear**, and **saturates at ~6 concurrent models
   (~100 img/s, 5× sequential)**. A 7th model adds latency (60 → 71 ms) with no
   throughput gain; an 8th exceeds the 8 GB budget. → `throughput_scaling.png`
3. **Batching beats same-model concurrency** (batch-4 = 83 vs. 4-concurrent = 74
   img/s). BUT a multi-disease *panel* uses **different** models, which **cannot be
   batched** — so concurrency is the only way to serve them together, and it still
   gives a 3.5–5× gain over sequential. → `power_efficiency.png`
4. **The memory wall is per-process, not model weights.** Weights are tiny (~32 MB
   each); the ceiling comes from each process's CUDA/cuDNN context (~0.8–1 GB). On
   8 GB that caps process-per-model concurrency at ~6–7. A shared-context design
   (CUDA streams / TensorRT) would push further — see *Future work*.
5. **Combining batching *and* concurrency reaches the GPU's ~180 img/s ceiling
   (97% util)** — which neither reaches alone (batching 167, concurrency 100). The
   best config is a **panel of 4 different disease models, each batching 4 images**
   (`C4b4_diff`): 180 img/s at 18 W, GPU saturated. This is the config the demo
   should showcase. → `saturation.png`, `regime_peak.png`

## TensorRT FP16 — the real edge-deployment path (tested)

Porting the DenseNet to a **TensorRT FP16 engine** (layer fusion + kernel
auto-tuning for this exact chip — *not* INT8 quantization, so accuracy is
preserved) transforms the box. → `pytorch_vs_tensorrt.png`

| Stage | Throughput | vs naive |
|---|---:|---:|
| PyTorch sequential (naive) | 21 img/s | 1× |
| PyTorch best (concurrent + batched) | 180 img/s | 9× |
| TensorRT FP16, single stream | 271 img/s | 13× |
| **TensorRT FP16, batched ×8** | **509 img/s** | **25×** |

- **Accuracy preserved:** TRT FP16 vs PyTorch FP32 pathology probabilities agree to
  within **0.86 percentage points** (max; 0.05 pp mean). FP16, not INT8 — no
  quantization accuracy risk.
- **TRT engines overlap in-process** where PyTorch CUDA streams did not: K=1→8
  concurrent engines scale 252→460 img/s, all in one process (one engine, ~30 MB
  per context — no memory wall). TRT issues one `execute` call per model instead of
  200 GIL-serialized kernel launches, so streams actually overlap.
- Consistent with the PyTorch findings, **batching one engine (509) still beats 8
  concurrent engines (460)** — batching remains the efficiency king.

Pipeline: `src/trt_export.py` (→ ONNX) → `trtexec --fp16` (→ engine) →
`src/trt_runner.py` (accuracy check + concurrent benchmark). Raw numbers in
`results/trt_bench.json`.

### INT8 quantization — the speed/accuracy trade-off, measured

Unlike FP16, INT8 quantization *can* move predictions — so we calibrated an INT8
engine (entropy calibration on 512 ChestMNIST images) and measured real AUROC on
2000 labeled test images. → `int8_tradeoff.png`

| Precision | Throughput (batch 8) | AUROC | Engine |
|---|---:|---:|---:|
| PyTorch (FP16 autocast) | 167 img/s | 0.7405 | — |
| TensorRT FP16 | 509 img/s | **0.7405** (same) | 14.9 MB |
| TensorRT INT8 | **1035 img/s** (2×, **50× naive**) | 0.6868 (**−0.054**) | 8.8 MB |

- **FP16 is a free lunch** — identical AUROC to PyTorch, 3× the speed.
- **INT8 doubles throughput again (~1035 img/s)** but costs **0.054 AUROC (7%)** —
  a real, non-trivial drop from post-training quantization. It's a **screening vs.
  diagnosis** decision: fine for high-volume triage, not for a final read.
  (Quantization-aware training or per-channel calibration could narrow the gap —
  future work.) Pipeline: `src/trt_int8.py`, `src/trt_eval_auroc.py`.

## Realistic mixed-architecture panel

Beyond the homogeneous scaling curves, we ran one **genuinely heterogeneous**
panel — different architectures *and* sizes, all detecting chest disease:
ResNet-50 (23.5 M, 512 px) + three DenseNet-121 variants (6.97 M, 224 px; NIH,
CheXpert, PadChest). Concurrent, batch 1, MPS.

| Model | Params / input | Throughput | Latency |
|---|---|---:|---:|
| ResNet-50 | 23.5 M / 512 px | 30 img/s | 33 ms |
| DenseNet-121 ×3 | 6.97 M / 224 px | 15 img/s each | 66 ms |
| **Aggregate** | | **75.5 img/s** | GPU **97%** |

**Counter-intuitive finding:** the bigger 512 px ResNet-50 is the *fastest* per
image (33 ms vs 66 ms). DenseNet-121's densely-connected design is a pile of small,
launch-heavy, bandwidth-bound kernels that contend more under concurrency; ResNet-50's
fewer, larger, compute-dense kernels run efficiently even at 4× the pixels. So a
"heavier" model is not automatically the bottleneck — and the box saturates its GPU
serving a real mixed panel with room to spare. (Aside: nearly all published
chest-X-ray disease models are DenseNet-121 — CheXNet made it the field standard —
so ResNet-50 is essentially the only *distinct* architecture available with real
chest-disease weights in torchxrayvision.)

## Spending spare capacity on robustness (TTA / ensemble)

The concurrency measurements showed the GPU has spare capacity. Can we spend it to
make each diagnosis more *reliable* instead of just faster? Tested on **real labeled
data** — ChestMNIST-224 (NIH ChestX-ray14, 2000 test images, auto-downloaded, no
Kaggle account needed) — with macro-AUROC over the 14 pathologies. → `tta_robustness.png`

| Method | AUROC | Δ |
|---|---:|---:|
| Single pass | 0.741 | — |
| TTA (5 augmented views) | 0.738 | −0.003 |
| **Ensemble (3 different-dataset DenseNets)** | **0.762** | **+0.022** |

- **Ensembling different models genuinely helps** (+0.022 AUROC) — averaging models
  trained on different datasets cancels individual errors.
- **Naive TTA does not help here** (−0.003, within noise) — consistent with the
  literature: mild augmentation averaging isn't automatically beneficial. An honest
  negative result, not hidden.
- **The cost is ~free:** 5 views (or 3 models) run as a **single batch** cost
  **49.2 ms vs 49.0 ms** for one pass — the spare capacity we measured absorbs the
  extra work. So on this box, the robustness gain is nearly latency-free.

Pipeline: `src/tta_experiment.py` (downloads ChestMNIST, runs single/TTA/ensemble,
computes AUROC + systems cost). Data in `results/tta_bench.json`.

## The clinic story

- One **$249 box** runs a **multi-disease chest X-ray panel locally** — no cloud,
  patient images never leave the building (privacy / GDPR / HIPAA-friendly).
- **No subscription:** one-time hardware pays back a comparable cloud GPU in
  ~4 months under a clinic duty cycle. → `cost_comparison.png`
- Different clinicians can ask **different questions at the same time** (pneumonia,
  cardiomegaly, effusion, the full 14-pathology panel) against the shared box, each
  getting sub-second answers — the `C*_diff` case measured here.

## How the parallelism works (plainly)

A GPU has one set of compute units. "Running models in parallel" means the
scheduler **interleaves** their kernels — a second model fills the gaps the first
leaves idle. Gains are **real but sublinear**, and that sublinear curve *is* the
finding, not a bug. Batching instead fuses many images into one efficient kernel
launch, which is why it wins for a single model — but it can't mix *different*
models, which is exactly where concurrency earns its place.

## Reproduce

Environment setup (Jetson-specific wheels + two library fixes) is documented in
`requirements.txt`. Then, on the board:

```bash
# Baselines (sequential + batched)
~/xray-venv/bin/python src/benchmark.py --configs S1,B2,B4,B8 --repeats 3

# Concurrency (same-model, different-model panel, ramp) under CUDA MPS
setsid bash scripts/run_concurrent.sh --repeats 3 --duration 8 \
    --same 2,4,5 --diff 2,4,5 --ramp 3,6,7 < /dev/null &

# Concurrent + batched (each of N models runs a batch of B)
setsid bash scripts/run_concurrent.sh --repeats 3 --duration 8 \
    --same 2,3,4 --diff 2,4 --ramp , --batch 4 < /dev/null &
```

Long board jobs must be launched **detached with `setsid`** (plain `nohup &` over
SSH gets SIGHUP'd when the channel closes). The script self-redirects to
`~/conc_run.log`.

Pull `results/raw/` back to a workstation and build the figures:

```bash
python analysis/make_figures.py     # writes results/figures/*.png
```

## Repo layout

```
src/
  models.py            # pretrained model registry (DenseNet variants + ResNet)
  utils.py             # preprocessing, input pool, CUDA-synced timing, percentiles
  power_logger.py      # tegrastats wrapper (VDD_IN power, GR3D_FREQ util, temp)
  runner_sequential.py # regime 1
  runner_batched.py    # regime 2
  runner_concurrent.py # regime 3 (multiprocessing + MPS, barrier-timed window)
  benchmark.py         # baseline orchestrator -> results/raw/*.json
  benchmark_concurrent.py  # concurrency orchestrator (+ MPS control)
scripts/run_concurrent.sh  # detached launcher (setsid-safe)
analysis/make_figures.py   # results/raw -> results/figures
results/{raw,figures}      # per-run JSON (gitignored) + committed PNGs
PLAN.md                    # full build spec
```

## Honesty guardrails

- Pretrained models' **published** accuracy only — no clinical validity claims.
- The $249 is a **hook**, not a hard constraint (it also runs on bigger boxes).
- "True parallel" on one GPU = interleaved kernels; gains are **sublinear** — stated
  plainly, because the sublinear curve is the result.

## CUDA streams: memory vs. throughput trade-off (tested)

We tested the obvious fix for the memory wall — run all N models in **one process,
one CUDA context, one stream each** (`src/runner_streams.py`). Result:

| N models | throughput | round latency | peak memory |
|---:|---:|---:|---:|
| 2 | 20 img/s | 99 ms | 93 MB |
| 8 | 20 img/s | 400 ms | 446 MB |
| 12 | 20 img/s | 604 ms | 680 MB |

- ✅ **Memory wall gone** — 12 models fit in 680 MB (vs. the process-per-model
  approach dying at 8 on 8 GB).
- ❌ **No throughput gain** — pinned at the single-model rate (~20 img/s) for *any*
  N; round latency grows strictly linearly. The models run **serially**: they are
  batch-1, launch-bound, and Python's GIL issues one model's kernels fully before
  the next, so nothing overlaps.

**Conclusion:** the two concurrency mechanisms trade off. **MPS + multiprocessing**
gives real throughput overlap (100 img/s at 6 models) but is memory-capped ~6–7;
**CUDA streams** scale to many models in tiny memory but serialize execution. For
this workload the throughput path is MPS-multiprocessing (or plain batching); streams
only help when you are memory-bound and throughput-tolerant.

## Future work

- **TensorRT the full different-models panel** (export every DenseNet variant +
  ResNet-50 to engines, run concurrently) — single-model TRT is done; the mixed
  panel on TRT is the natural next step.
- **Quantization-aware training / per-channel INT8** to recover the 0.054 AUROC
  that post-training INT8 lost, while keeping the 2× speedup.
- **TTA / ensembling**: spend spare concurrent capacity on diagnostic robustness
  (AUROC) rather than only throughput.
- **Live multi-clinician demo** (Section 9 of PLAN.md) with a real-time
  throughput/power readout.
