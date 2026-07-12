# XP4 — Realistic mixed-architecture panel

A genuinely heterogeneous panel — different architectures *and* sizes, all detecting
chest disease: **ResNet-50 (23.5 M, 512 px) + 3 DenseNet-121 (6.97 M, 224 px)**,
concurrent, batch 1.

## Result
| Model | Params / input | Throughput | Latency |
|---|---|---:|---:|
| ResNet-50 | 23.5 M / 512 px | 30 img/s | 33 ms |
| DenseNet-121 ×3 | 6.97 M / 224 px | 15 img/s each | 66 ms |
| **Aggregate** | | **75.5 img/s** | GPU 97% |

**Counter-intuitive: the bigger 512 px ResNet-50 is the *fastest* per image.** It's
faster even *standalone*, and GPU utilisation shows exactly why (`probe_standalone.py`):

| Model (batch 1, alone) | Params / input | Latency | **GPU util** |
|---|---|---:|---:|
| DenseNet-121 | 7.0 M / 224 px | 49.1 ms | **18 %** |
| ResNet-50 | 23.5 M / 512 px | **23.9 ms** | **94 %** |

At batch 1, latency is set by **kernel-launch overhead and the dependency chain, not
FLOPs.** DenseNet-121's ~120 densely-connected layers become hundreds of tiny kernel
launches, each doing trivial work — so it's *launch-bound* and leaves **82 % of the GPU
idle**. ResNet-50 has fewer, larger, compute-dense convolutions that keep the GPU busy
(**94 %**), so despite ~7× the FLOPs it finishes in half the time. "Bigger" ≠ slower;
at low batch, **kernel granularity beats parameter count.** (This 18 % idle is also why
batching gave DenseNet an 8× jump in XP1 — batching fills those idle cycles; ResNet at
94 % is already near-saturated and gains far less.)

(Aside: nearly all published chest-disease models are DenseNet-121 — CheXNet made it
the standard — so ResNet-50 is the only distinct architecture with real chest weights.)

## Run
```bash
setsid bash run_panel.sh                 # the heterogeneous panel (3 runs)
~/xray-venv/bin/python probe_standalone.py   # standalone latency + GPU-util comparison
```

## Files
- `run_panel.sh` — runnable entry point for the panel. **The panel itself is a
  *config* of XP2's concurrency engine** (`benchmark_concurrent.py --panel "<models>"`),
  not a new algorithm — so this is a thin wrapper, not a reimplementation. That's why
  XP4 has no core runner of its own (same for XP3, which is XP2 `--batch`).
- `probe_standalone.py` — the per-model batch-1 latency + GPU-utilisation measurement
  behind the "bigger model is faster" result.
