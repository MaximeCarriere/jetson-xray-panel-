# XP2 — Concurrency (multiprocessing + CUDA MPS) & the memory wall

Run **N models at once**, one process each, under CUDA MPS. This is the heart of
the project: a multi-disease *panel* uses different models that can't be batched
together, so concurrency is the only way to serve them at once.

## Result (mean over 3 runs, ±1 SE)
| N concurrent | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| img/s | 19.9 | 39.4 | 57.3 | 73.8 | 89.3 | **101.0** | 100.4 | ✗ |
| ±SE | 0.0 | 0.1 | 0.4 | 0.4 | 0.4 | 0.3 | 0.1 | — |
| latency ms | 50 | 50 | 53 | 54 | 56 | 60 | **71** | — |

- Real but **sublinear**; **saturates at N≈6** (~100 img/s, 5×). N=7 adds latency, no gain.
- **N=8 hits a memory wall** — 8 per-process CUDA contexts (~1 GB each) exceed the
  8 GB board and thrash. The ceiling is per-process *context* memory, not weights.
- Same-model ≈ different-model throughput (the GPU does the same work) — but only the
  different-model panel *needs* concurrency.

![throughput scaling](../../results/figures/throughput_scaling.png)
![power and efficiency](../../results/figures/power_efficiency.png)

## Run
```bash
setsid bash run_concurrent.sh --repeats 3 --duration 8 --same 2,4 --diff 2,4 --ramp 3,5,6
```

## Files
`runner_concurrent.py` (process-per-model, barrier-timed window) ·
`benchmark_concurrent.py` (orchestrator, MPS control) · `probe_ramp.py` (ceiling probe) ·
`run_concurrent.sh` (detached launcher). Also drives XP3 and XP4.
