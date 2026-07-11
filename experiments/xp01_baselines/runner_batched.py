"""Regime 2 — batched baseline (Section 4 of PLAN.md).

The SMART single-model way: stack N images into one tensor and run a single
forward pass. The GPU processes them together, amortizing per-call overhead and
filling more of the compute — so throughput rises with batch size until the GPU
saturates. This is the honest baseline that concurrency must be compared against
(for one disease, batching is usually the most efficient option).

Note the trade-off batching cannot solve: a batch must go through ONE model, so
a multi-model *panel* (different diseases -> different weights) cannot be served
by a single batched pass. That is what the concurrent regime is for.
"""
from __future__ import annotations

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
from utils import timed_inference


def run(model, pool, batch_size: int, n_batches: int,
        warmup: int = 10, use_autocast: bool = True) -> dict:
    """Run ``n_batches`` forward passes of ``batch_size`` images each.

    ``per_call_ms`` here is *per-batch* latency (the wait for a whole batch to
    finish); ``batch_size`` is recorded so this stays interpretable downstream.
    """
    n = pool.shape[0]
    assert n >= batch_size, "input pool smaller than batch size"
    state = {"i": 0}

    def step() -> None:
        start = (state["i"] * batch_size) % n
        x = pool[start:start + batch_size]
        if x.shape[0] < batch_size:                     # wrap around the pool
            x = torch.cat([x, pool[:batch_size - x.shape[0]]], dim=0)
        state["i"] += 1
        with torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)

    wall_s, per_call_ms = timed_inference(step, n_batches, warmup)
    return {
        "wall_s": wall_s,
        "per_call_ms": per_call_ms,         # per-BATCH latency
        "n_images": n_batches * batch_size,
        "batch_size": batch_size,
    }
