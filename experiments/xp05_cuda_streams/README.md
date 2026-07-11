# XP5 — CUDA streams: memory vs throughput trade-off

Can we break XP2's memory wall by running all N models in **one process, one CUDA
context, one stream each**? Test result: it fixes memory but not throughput.

## Result (mean ± SE over 3 runs)
| N models | throughput | round latency | peak memory |
|---:|---:|---:|---:|
| 2 | 15.9 ± 0.1 img/s | 126 ms | 112 MB |
| 8 | 19.4 ± 0.1 img/s | 413 ms | 560 MB |
| 12 | 18.3 ± 1.2 img/s | 661 ms | **680 MB** |

![streams](../../results/figures/streams.png)

- ✅ **Memory wall gone** — 12 models in 680 MB (one context).
- ❌ **No throughput gain** — pinned at the single-model rate (~16–19 img/s) for any N;
  the models run **serially**. They are batch-1, launch-bound, and Python's GIL issues
  one model's kernels fully before the next, so nothing overlaps.

**Conclusion:** MPS+multiprocessing (XP2) gives throughput but is memory-capped ~6;
CUDA streams give memory but serialize. TensorRT (XP6) fixes both.

## Run
```bash
setsid bash run_streams.sh 2 4 6 8 10 12
```

## Files
`runner_streams.py` · `run_streams.sh`.
