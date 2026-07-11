"""Benchmark orchestrator (Sections 4-6 of PLAN.md).

Runs each configuration with proper measurement hygiene — warm-up, steady-state
timing, continuous power logging aligned to the run window, and 3x repeats — then
writes one JSON record per run to results/raw/ in the Section-6 schema.

Usage on the board:

    ~/xray-venv/bin/python ~/jetson-xray-panel/experiments/xp01_baselines/benchmark.py \
        --configs S1,B2,B4,B8 --repeats 3 --n-images 200

This module covers the Week-1 baselines (sequential + batched). The concurrent
regime lives in runner_concurrent.py and is orchestrated separately.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
import models
import runner_batched
import runner_sequential
import utils
from power_logger import PowerLogger

POWER_MODE = "MAXN_SUPER"      # locked mode for all runs (see PLAN.md)
RUNTIME = "pytorch"
WORKHORSE = "densenet121-res224-all"   # the single-model baseline model
RESULTS_DIR = os.path.expanduser("~/jetson-xray-panel/results/raw")


def _assemble_record(config: str, regime: str, model_names: list[str],
                     result: dict, power: dict, repeat_index: int) -> dict:
    """Turn timing + power primitives into a schema-valid results record."""
    thr = result["n_images"] / result["wall_s"]          # images / second
    lat = utils.latency_stats(result["per_call_ms"])
    p_mean = power["power_w"]["mean"]
    return {
        "config": config,
        "regime": regime,
        "n_models": len(model_names),
        "model_names": model_names,
        "batch_size": result["batch_size"],
        "runtime": RUNTIME,
        "power_mode": POWER_MODE,
        "n_inferences": result["n_images"],
        "throughput_ips": round(thr, 2),
        "latency_ms": {k: round(v, 3) for k, v in lat.items()},
        "gpu_util_pct": power["gpu_util_pct"],
        "power_w": power["power_w"],
        "compute_power_w": power["compute_power_w"],
        "temp_c_peak": power["temp_c_peak"],
        "mem_mb_peak": round(utils.peak_mem_mb(), 1),
        "throughput_per_watt": round(thr / p_mean, 3) if p_mean else None,
        "repeat_index": repeat_index,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
    }


def _measure(config: str, regime: str, model_names: list[str],
             run_fn, repeat_index: int) -> dict:
    """Wrap one run in power logging + peak-memory tracking, return the record."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with PowerLogger(interval_ms=100) as p:
        t0 = time.perf_counter()
        result = run_fn()
        t1 = time.perf_counter()
    power = p.summary(t0, t1)
    rec = _assemble_record(config, regime, model_names, result, power, repeat_index)
    _write(rec)
    print(f"  [{config} r{repeat_index}] {rec['throughput_ips']:>7.2f} img/s | "
          f"lat {rec['latency_ms']['mean']:.2f}ms | {rec['power_w']['mean']}W | "
          f"{rec['throughput_per_watt']} img/s/W | GPU {rec['gpu_util_pct']}%")
    return rec


def _write(rec: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fn = f"{rec['config']}_r{rec['repeat_index']}_{int(time.time() * 1000)}.json"
    with open(os.path.join(RESULTS_DIR, fn), "w") as f:
        json.dump(rec, f, indent=2)


# --------------------------------------------------------------------------- #
# Config runners
# --------------------------------------------------------------------------- #

def run_config(config: str, model, pool, n_images: int, repeat_index: int) -> dict:
    """Dispatch a single named config (S1 / B2 / B4 / B8) for one repeat."""
    if config == "S1":
        return _measure(
            config, "sequential", [WORKHORSE], repeat_index=repeat_index,
            run_fn=lambda: runner_sequential.run(model, pool, n_iter=n_images),
        )
    if config.startswith("B"):
        bs = int(config[1:])
        n_batches = max(25, n_images // bs)
        return _measure(
            config, "batched", [WORKHORSE], repeat_index=repeat_index,
            run_fn=lambda: runner_batched.run(model, pool, batch_size=bs,
                                              n_batches=n_batches),
        )
    raise ValueError(f"unknown config {config!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="S1,B2,B4,B8",
                    help="comma-separated config names")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--image-dir", default=None,
                    help="dir of real X-rays; if absent, synthetic images are used")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    utils.seed_everything(0)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"Device: {torch.cuda.get_device_name(0)} | power mode {POWER_MODE}")
    print(f"Configs: {configs} x{args.repeats} repeats, ~{args.n_images} images each\n")

    # Single-model baselines all use the workhorse model + a shared input pool.
    size = models.input_size_for(WORKHORSE)
    model = models.load_model(WORKHORSE, device="cuda")
    max_bs = max([int(c[1:]) for c in configs if c.startswith("B")] + [1])
    pool = utils.build_input_pool(size=size, n=max(64, max_bs), image_dir=args.image_dir)
    print(f"Model {WORKHORSE} loaded; input pool {tuple(pool.shape)}"
          f" ({'real' if args.image_dir else 'synthetic'} images)\n")

    for config in configs:
        print(f"== {config} ==")
        for r in range(1, args.repeats + 1):
            run_config(config, model, pool, args.n_images, repeat_index=r)
    print(f"\nDone. Records in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
