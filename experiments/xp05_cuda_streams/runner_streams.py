"""Regime 3b — concurrent models via CUDA streams in ONE process.

The multiprocessing+MPS runner hits a memory wall at ~N=8 because each process
carries its own ~1 GB CUDA/cuDNN context. This runner instead loads all N models
in a SINGLE process and gives each its own CUDA stream. There is only one CUDA
context (~1 GB) plus the tiny model weights (~32 MB each), so memory scales far
better — the question is whether throughput holds up, since Python launches
kernels sequentially (GIL) even though the GPU can overlap work across streams.

Run a scaling sweep (single process, touches CUDA directly):

    ~/xray-venv/bin/python runner_streams.py 2 4 6 8 10 12
"""
from __future__ import annotations

import sys
import time

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
import models
import utils


def run_streams(model_names: list[str], duration: float = 8.0, warmup: int = 15,
                batch_size: int = 1) -> dict:
    """Run all models concurrently via one CUDA stream each, for ``duration`` s."""
    dev = "cuda"
    loaded = [models.load_model(n, dev) for n in model_names]
    streams = [torch.cuda.Stream() for _ in model_names]
    pools = [utils.build_input_pool(models.input_size_for(n), max(8, batch_size), device=dev)
             for n in model_names]
    torch.cuda.reset_peak_memory_stats()

    def one_round() -> None:
        # Issue every model's forward on its own stream, then wait for all. The
        # launches are sequential (GIL) but execution overlaps across streams.
        for model, stream, x in zip(loaded, streams, pools):
            with torch.cuda.stream(stream):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    model(x[:batch_size])
        torch.cuda.synchronize()

    for _ in range(warmup):
        one_round()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    rounds, lats = 0, []
    while time.perf_counter() - t0 < duration:
        ts = time.perf_counter()
        one_round()
        lats.append((time.perf_counter() - ts) * 1000.0)   # per-round latency
        rounds += 1
    elapsed = time.perf_counter() - t0

    n = len(model_names)
    return {
        "n_models": n,
        "batch_size": batch_size,
        "n_images": rounds * n * batch_size,
        "wall_s": elapsed,
        "throughput_ips": (rounds * n * batch_size) / elapsed,
        "round_latency_ms_mean": sum(lats) / len(lats) if lats else None,
        "mem_mb_peak": utils.peak_mem_mb(),
    }


if __name__ == "__main__":
    import json
    from stats import agg

    ns = [int(x) for x in sys.argv[1:]] or [2, 4, 6, 8, 10, 12]
    REPEATS = 3
    workhorse = "densenet121-res224-all"
    rows = []
    print(f"{'N':>3}  {'img/s (mean±SE)':>18}  {'round ms':>9}  {'peak MB':>8}")
    for n in ns:
        try:
            ips, lat, mem = [], [], []
            for _ in range(REPEATS):
                r = run_streams([workhorse] * n, duration=6.0)
                ips.append(r["throughput_ips"]); lat.append(r["round_latency_ms_mean"])
                mem.append(r["mem_mb_peak"])
            a = agg(ips)
            rows.append({"n": n, "ips_mean": round(a["mean"], 1), "ips_se": round(a["se"], 2),
                         "round_ms": round(sum(lat) / len(lat), 1),
                         "mem_mb": round(sum(mem) / len(mem))})
            print(f"{n:>3}  {a['mean']:>11.1f} ± {a['se']:<4.2f}  "
                  f"{rows[-1]['round_ms']:>9.1f}  {rows[-1]['mem_mb']:>8}")
        except Exception as e:
            print(f"{n:>3}  FAILED: {type(e).__name__}: {e}")
            break
    with open("/home/a/jetson-xray-panel/results/streams_bench.json", "w") as f:
        json.dump({"repeats": REPEATS, "rows": rows}, f, indent=2)
    print("wrote results/streams_bench.json")
