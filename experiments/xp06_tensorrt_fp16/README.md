# XP6 — TensorRT FP16

PyTorch runs the model as a *general* framework: it launches each layer as a separate
GPU kernel through Python, which is flexible but wasteful. **TensorRT** is NVIDIA's
inference compiler: it takes the trained network and, ahead of time, **fuses layers**
(e.g. conv+batchnorm+ReLU become one kernel), **picks the fastest kernel implementation
for this exact chip** (auto-tuning against the Orin's actual GPU), and emits a single
compiled **engine**. Same math, far fewer kernel launches and memory round-trips.

We use **FP16** (16-bit float), *not* INT8 — FP16 halves memory bandwidth and doubles
math throughput but keeps essentially full precision, so **accuracy is preserved**.
(The INT8 story — 2× faster again but a real accuracy hit — is XP7.)

## What "single stream" means

A **CUDA stream** is a single ordered queue of GPU work. **TensorRT FP16, single stream**
means: **one engine, one stream, batch size 1, images fed one at a time** — no batching,
no concurrency. It's the pure, apples-to-apples successor of *PyTorch sequential* (also
one image at a time), so it isolates the speedup from **compilation alone** (271 vs 20
img/s = 14×) *before* we stack batching or concurrent streams on top. The later rows add
those: **batched ×8** feeds 8 images per call, and the concurrent-streams sweep (below)
runs several engines on separate streams that genuinely overlap in one process.

## Result
Throughput = mean ± SE over 3 spaced runs. Macro-AUROC = 14-pathology average on 2000
labeled ChestMNIST images (1000-sample bootstrap SE).

| Stage | Throughput | vs naive | Macro-AUROC |
|---|---:|---:|---:|
| PyTorch sequential (FP32) | 19.9 img/s | 1× | 0.7405 ± 0.0134 |
| PyTorch best (concurrent + batched) | 180 img/s | 9× | 0.7405 ± 0.0134 |
| TensorRT FP16, single stream (batch 1) | 271.1 ± 0.4 img/s | 14× | 0.7405 ± 0.0134 |
| **TensorRT FP16, batched ×8** | **507.7 ± 0.6 img/s** | **26×** | **0.7405 ± 0.0134** |

- **Accuracy is unchanged across every row** — that's the point of the column: throughput
  climbs **26×** while macro-AUROC stays flat at **0.7405**. Batching, concurrency, and
  FP16 change *how fast* the model runs, not *what it computes*. FP16 vs PyTorch FP32
  pathology probabilities agree to within **0.86 pp** (identical AUROC). See
  `trt_eval_auroc.py`.
- **TRT engines overlap in-process** (252→460 img/s, K=1..8 streams) where PyTorch CUDA
  streams did not, with no memory wall — one execute call per model, not 200
  GIL-serialized kernel launches.

![pytorch vs tensorrt](../../results/figures/pytorch_vs_tensorrt.png)

## Run
```bash
~/xray-venv/bin/python trt_export.py densenet121-res224-all ~/densenet_all.onnx --batch 1
/usr/src/tensorrt/bin/trtexec --onnx=~/densenet_all.onnx --fp16 --saveEngine=~/densenet_all.engine \
    --minShapes=image:1x1x224x224 --optShapes=image:4x1x224x224 --maxShapes=image:8x1x224x224
~/xray-venv/bin/python ../../lib/trt_runner.py ~/densenet_all.engine   # accuracy + concurrent test
```

## Files
`trt_export.py` (→ ONNX) · `trt_eval_auroc.py` (AUROC on labeled data). Runtime
`lib/trt_runner.py`. Data `../../results/trt_bench.json`.
