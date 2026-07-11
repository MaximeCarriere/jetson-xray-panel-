"""Regime 1 — sequential baseline (Section 4 of PLAN.md).

The naive way: push images through ONE model ONE AT A TIME. This is the floor we
compare everything else against. It underuses the GPU (each tiny forward pass
leaves most of the compute idle), which is exactly why batching and concurrency
can beat it.
"""
from __future__ import annotations

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
from utils import timed_inference


def run(model, pool, n_iter: int, warmup: int = 10, use_autocast: bool = True) -> dict:
    """Run ``n_iter`` single-image forward passes, cycling through ``pool``.

    Returns timing primitives; the orchestrator turns these into the results JSON.
    """
    n = pool.shape[0]
    state = {"i": 0}

    def step() -> None:
        k = state["i"] % n
        x = pool[k:k + 1]                       # one image: (1, 1, H, W)
        state["i"] += 1
        with torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)

    wall_s, per_call_ms = timed_inference(step, n_iter, warmup)
    return {
        "wall_s": wall_s,
        "per_call_ms": per_call_ms,     # per-image latency (batch size 1)
        "n_images": n_iter,
        "batch_size": 1,
    }
