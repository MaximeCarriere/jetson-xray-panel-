# XP5 — CUDA streams: memory vs throughput trade-off

Can we break XP2's memory wall by running all N models in **one process, one CUDA
context, one stream each**? Test result: it fixes memory but not throughput.

## Result
| N models | throughput | round latency | peak memory |
|---:|---:|---:|---:|
| 2 | 20 img/s | 99 ms | 93 MB |
| 8 | 20 img/s | 400 ms | 446 MB |
| 12 | 20 img/s | 604 ms | **680 MB** |

- ✅ **Memory wall gone** — 12 models in 680 MB (one context).
- ❌ **No throughput gain** — pinned at the single-model rate (~20 img/s) for any N;
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
