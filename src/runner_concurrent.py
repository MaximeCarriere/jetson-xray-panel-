"""Regime 3 — concurrent multi-model inference (Section 4 of PLAN.md).

This is the heart of the project. We run N models *at the same time*, one OS
process each, and measure the aggregate throughput the single GPU delivers.

Why multiple processes (not threads or one process)?
  * Python's GIL and a single CUDA context serialize CPU-side work within one
    process, which masks real GPU concurrency.
  * Separate processes each get their own CUDA context; the GPU's scheduler then
    interleaves their kernels — a second model's work fills the gaps the first
    leaves idle. That interleaving is the "concurrency" we are measuring.
  * With CUDA MPS enabled, those contexts can even run *spatially* at once rather
    than being time-sliced. MPS is toggled by the orchestrator via env vars; this
    runner is agnostic to it.

Measurement model: every worker runs SYNCHRONOUSLY (it waits for each inference,
like a real clinician waiting for a result) for a fixed wall-clock DURATION, and
we count how many inferences it completed. Aggregate throughput = total
inferences / duration. A barrier makes all workers start their timed window
together, so the measured window is genuinely concurrent.
"""
from __future__ import annotations

import time

import torch
import torch.multiprocessing as mp

import models
import utils


def _worker(rank: int, model_name: str, duration: float, warmup: int,
            use_autocast: bool, batch_size: int, barrier, ret_q) -> None:
    """One model in one process: warm up, sync at the barrier, then run timed.

    Each timed step runs ``batch_size`` images through the model, so we can
    combine concurrency (N processes) with batching (B images each).
    """
    try:
        torch.cuda.init()
        utils.seed_everything(rank)
        size = models.input_size_for(model_name)
        model = models.load_model(model_name, device="cuda")
        # Private pool must hold at least one batch.
        x = utils.build_input_pool(size=size, n=max(8, batch_size), device="cuda")
        torch.cuda.reset_peak_memory_stats()

        def one() -> None:
            with torch.no_grad():
                if use_autocast:
                    with torch.autocast("cuda", dtype=torch.float16):
                        model(x[:batch_size])
                else:
                    model(x[:batch_size])

        for _ in range(warmup):
            one()
        torch.cuda.synchronize()

        # All workers AND the parent meet here, so the parent can bracket power
        # logging to exactly the timed window (excluding spawn/load/warmup).
        barrier.wait(timeout=180)
        t0 = time.perf_counter()
        lats, count = [], 0
        while time.perf_counter() - t0 < duration:
            ts = time.perf_counter()
            one()
            torch.cuda.synchronize()         # synchronous per-request latency
            lats.append((time.perf_counter() - ts) * 1000.0)
            count += 1
        elapsed = time.perf_counter() - t0

        ret_q.put({
            "rank": rank,
            "model_name": model_name,
            "count": count,                       # timed steps (each = batch_size images)
            "images": count * batch_size,
            "elapsed_s": elapsed,
            "latencies_ms": lats,                 # per-BATCH latency
            "peak_mem_mb": utils.peak_mem_mb(),
        })
    except Exception as e:                    # surface worker failures to the parent
        ret_q.put({"rank": rank, "error": f"{type(e).__name__}: {e}"})


def run_concurrent(model_names: list[str], duration: float = 8.0,
                   warmup: int = 15, use_autocast: bool = True,
                   batch_size: int = 1) -> dict:
    """Spawn one process per model, run them concurrently for ``duration`` seconds.

    ``batch_size`` > 1 combines concurrency with batching (N models, each running
    B images per step).

    IMPORTANT: the parent must NOT have initialized CUDA (spawn gives fresh
    processes, but we keep the parent CUDA-free so PowerLogger etc. stay clean).
    Returns aggregate + per-worker metrics.
    """
    ctx = mp.get_context("spawn")
    n = len(model_names)
    barrier = ctx.Barrier(n + 1)                # +1 for the parent (window bracketing)
    ret_q = ctx.Queue()

    procs = [
        ctx.Process(target=_worker,
                    args=(i, model_names[i], duration, warmup, use_autocast,
                          batch_size, barrier, ret_q))
        for i in range(n)
    ]
    for p in procs:
        p.start()

    # Wait until every worker has loaded + warmed up; the barrier releases us and
    # the workers together, so [t_run_start, t_run_end] is the concurrent window.
    try:
        barrier.wait(timeout=180)
    except Exception as e:
        for p in procs:
            p.terminate()
        raise RuntimeError(f"workers did not reach start barrier: {e}")
    t_run_start = time.perf_counter()

    results = [ret_q.get() for _ in range(n)]   # collect before join (queue drain)
    t_run_end = time.perf_counter()
    for p in procs:
        p.join()

    errors = [r for r in results if "error" in r]
    if errors:
        raise RuntimeError(f"worker(s) failed: {[e['error'] for e in errors]}")

    total_images = sum(r["images"] for r in results)
    all_lats = [l for r in results for l in r["latencies_ms"]]
    return {
        "wall_s": duration,
        "t_run_start": t_run_start,     # parent-clock bounds of the concurrent window
        "t_run_end": t_run_end,
        "per_call_ms": all_lats,        # per-BATCH latency
        "n_images": total_images,
        "batch_size": batch_size,
        "per_worker": [
            {"rank": r["rank"], "model": r["model_name"],
             "throughput_ips": round(r["images"] / r["elapsed_s"], 2),
             "latency_ms_mean": round(sum(r["latencies_ms"]) / len(r["latencies_ms"]), 2)
             if r["latencies_ms"] else None,
             "peak_mem_mb": round(r["peak_mem_mb"], 1)}
            for r in sorted(results, key=lambda r: r["rank"])
        ],
        "mem_mb_peak_sum": round(sum(r["peak_mem_mb"] for r in results), 1),
    }


if __name__ == "__main__":
    # Quick validation: does concurrency actually overlap? Compare aggregate
    # throughput at N=1,2,4 identical models. If N=2 >> N=1, the GPU is truly
    # running them concurrently (filling idle gaps).
    WORKHORSE = "densenet121-res224-all"
    for n in (1, 2, 4):
        r = run_concurrent([WORKHORSE] * n, duration=6.0)
        print(f"N={n}: aggregate {r['n_images'] / r['wall_s']:6.1f} img/s "
              f"| per-worker {[w['throughput_ips'] for w in r['per_worker']]}")
