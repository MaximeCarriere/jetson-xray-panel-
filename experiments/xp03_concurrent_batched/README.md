# XP3 — Concurrent + batched (reaching the GPU ceiling)

Combine both forms of parallelism: **N model copies, each running a batch of B**.
Neither alone saturates the GPU — together they do.

## Result (mean ± SE over 3 runs)
| Config | img/s | GPU |
|---|---:|---:|
| Concurrent only (6 × 1) | 101.0 ± 0.3 | 79 % |
| Batched only (1 × 8) | 160.5 ± 0.1 | 83 % |
| **Concurrent + batched (4 × 4)** | **179.3 ± 0.7** | **97 %** |

The batch-4 sweep in full: `C2b4` 157.2 ± 0.7 (89 %), `C3b4` 177.5 ± 0.5 (96 %),
`C4b4` 179.3 ± 0.7 (97 %); the batch-2 sweep: `C2b2` 80.0 ± 0.4, `C4b2` 146.4 ± 0.7,
`C6b2` 163.8 ± 0.4. Run-to-run variance is under 0.5 % (error bars in the figures are
therefore tiny — that *is* the reproducibility result).

**The device's real ceiling is ~180 img/s at 97 % GPU**, reached by combining batching
and concurrency — neither reaches it alone (batching plateaus ~160, pure concurrency
~100). The winning config is 4 model copies each batching 4 images.

![reaching the ceiling](../../results/figures/saturation.png)
![peak by strategy](../../results/figures/regime_peak.png)

## Run
```bash
setsid bash run_batched.sh     # batch-4 and batch-2 sweeps, 3 repeats each
```

## Files
- `run_batched.sh` — runnable entry point. **The batched-concurrent regime is a
  *config* of XP2's concurrency engine** (`benchmark_concurrent.py --batch B`), not a
  new algorithm, so this is a thin wrapper rather than a reimplementation — which is
  why XP3 has no core runner of its own (same as XP4).
