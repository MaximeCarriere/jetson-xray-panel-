"""Why is the 'bigger' ResNet-50 faster than DenseNet-121 at batch 1?

Measure each model ALONE at batch 1: latency, throughput, and — the telling number —
GPU utilisation. A launch/latency-bound model leaves the GPU mostly idle; a
compute-efficient one keeps it busy.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

import torch

import models
from power_logger import PowerLogger

for name in ["densenet121-res224-nih", "resnet50-res512-all"]:
    size = models.input_size_for(name)
    n_params = None
    m = models.load_model(name, "cuda")
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.rand(1, 1, size, size, device="cuda")

    def one():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            m(x)

    for _ in range(20):
        one()
    torch.cuda.synchronize()
    with PowerLogger(interval_ms=100) as p:
        t0 = time.perf_counter()
        N = 200
        for _ in range(N):
            one()
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        s = p.summary(t0, t0 + dt)
    print(f"{name:28s} {n_params:5.1f}M params, {size}px | "
          f"{dt/N*1000:5.1f} ms/img | {N/dt:5.1f} img/s | "
          f"GPU {s['gpu_util_pct']:.0f}% | {s['power_w']['mean']} W", flush=True)
