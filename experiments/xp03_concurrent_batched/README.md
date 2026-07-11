# XP3 — Concurrent + batched (reaching the GPU ceiling)

Combine both forms of parallelism: **N models, each running a batch of B**. Neither
alone saturates the GPU — together they do.

## Result
| Config | img/s | GPU |
|---|---:|---:|
| Concurrent only (6×1) | 100 | 78% |
| Batched only (1×8) | 167 | 83% |
| **Concurrent + batched (4×4)** | **180** | **97%** |

The device's real ceiling is ~180 img/s at 97% GPU. `C4b4_diff` — 4 *different*
disease models each batching 4 images — is both the product story (a real panel)
and the highest-throughput config.

![reaching the ceiling](../../results/figures/saturation.png)
![peak by strategy](../../results/figures/regime_peak.png)

## Run (uses XP2's orchestrator with --batch)
```bash
setsid bash ../xp02_concurrency/run_concurrent.sh --repeats 3 --duration 8 \
    --same 2,3,4 --diff 2,4 --ramp , --batch 4
```

## Files
No new code — uses `../xp02_concurrency/benchmark_concurrent.py --batch B`.
