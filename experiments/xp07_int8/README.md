# XP7 — INT8 quantization (speed vs accuracy)

Unlike FP16, INT8 *can* move predictions — so we calibrate on real images and
measure the real AUROC cost.

## Result
| Precision | Throughput (batch 8) | AUROC (± bootstrap SE) | Engine |
|---|---:|---:|---:|
| TensorRT FP16 | 509 img/s | **0.7405 ± 0.0134** (= PyTorch) | 14.9 MB |
| TensorRT INT8 | **1035 img/s** (2×, 52× naive) | 0.6868 ± 0.0151 (**−0.054**) | 8.8 MB |

INT8 doubles throughput but costs **0.054 AUROC (7%)** — a **screening vs diagnosis**
call. FP16 remains the safe default. (QAT / per-channel INT8 could narrow the gap.)
AUROC re-verified on 2000 labeled ChestMNIST images; the point-estimate drop (0.054)
is ~3.6× the per-estimate SE, though a paired test would give tighter significance.

![int8 trade-off](../../results/figures/int8_tradeoff.png)

## Run
```bash
~/xray-venv/bin/python trt_int8.py ~/densenet_nih.onnx ~/densenet_nih_int8.engine
~/xray-venv/bin/python ../xp06_tensorrt_fp16/trt_eval_auroc.py ~/densenet_nih_int8.engine
```

## Files
`trt_int8.py` (calibrated INT8 build, torch-tensor calib buffer). Eval via XP6's
`trt_eval_auroc.py`. Data `../../results/trt_bench.json` (`int8` block).
