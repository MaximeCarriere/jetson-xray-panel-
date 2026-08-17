# XP16 — Compilation, precision matched

XP6 reported TensorRT FP16 against a PyTorch baseline it labelled "(FP32)", and
quoted accuracy from a different model than the one it timed. This experiment
removes both confounds so the compilation speedup can be stated on its own.

Everything below is **one model** (`densenet121-res224-nih`), **batch 8**,
`op_threshs=None` on both sides (the exported engine drops `op_norm`, see
`xp06/trt_export.py`), and the **same 2000 ChestMNIST-224 test images**.

## Result (mean ± SE over 3 runs)

| Config | Throughput | vs PyTorch FP16 | macro-AUROC |
|---|---:|---:|---:|
| PyTorch FP32 | 139.2 ± 0.2 img/s | 0.82× | 0.7405 ± 0.0134 |
| **PyTorch FP16 (autocast)** | **169.5 ± 0.1 img/s** | **1×** | **0.7405 ± 0.0134** |
| PyTorch FP16 (`model.half()`) | 184.5 ± 0.0 img/s | 1.09× | 0.5006 ± 0.0118 ⚠ |
| **TensorRT FP16** | **499.1 ± 0.9 img/s** | **2.94×** | **0.7405 ± 0.0134** |

**Compilation alone, at matched FP16, is 2.9×.** Accuracy is bit-for-bit
indistinguishable at four decimal places across every valid row.

## Two things this corrects

1. **The old 3.2× (507.7 / 160.5) was not precision matched.** `benchmark.py`
   calls `runner_batched.run()` without `use_autocast`, which defaults to
   `True` — so the 160.5 baseline was *already* FP16 autocast, while the AUROC
   reference (`pytorch_fp32`) was FP32. The two numbers described different
   runs. It also compared throughput on `-all` weights against accuracy on
   `-nih` weights.
2. **`model.half()` is not a usable FP16 path for this model.** It is the
   fastest PyTorch row at 184.5 img/s, and it is worthless: macro-AUROC 0.5006
   is chance. torchxrayvision scales inputs to ±1024 and pure-FP16 weights lose
   the model's discrimination (outputs stay finite, so nothing crashes — it
   just silently stops working). `lib/models.py` already warns about this for
   `op_norm`; the accuracy collapse is the bigger reason. Use `torch.autocast`.

## Run (on the board)
```bash
~/xray-venv/bin/python bench_compile_clean.py   # throughput + accuracy
~/xray-venv/bin/python acc_check.py             # accuracy across all 4 precisions
```
Both were run from `/home/a/jetson-xray-panel` (the board copy is flat, so the
`sys.path` inserts point at the top level plus `lib/`).

## Files
`bench_compile_clean.py` (throughput + AUROC, writes `results.json`) ·
`acc_check.py` (the four-way accuracy comparison, including the `model.half()`
failure) · `results.json` (raw throughput records).

Feeds the "Compilation" post on kernwerk.org.
