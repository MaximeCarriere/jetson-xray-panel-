"""Concurrent-regime orchestrator (Sections 4-6 of PLAN.md).

Runs the concurrency matrix and writes Section-6 JSON records, with the same
measurement hygiene as benchmark.py (warm-up in each worker, 3x repeats, power
logging aligned to the concurrent window). CUDA MPS is enabled by default (the
recommended primary method); pass --no-mps to measure plain time-slicing.

  * same-model  C{N}_same : N identical copies -> compare against batching (H3)
  * diff-model  C{N}_diff : N different models  -> the diagnostic panel (H4)
  * ramp        RAMP{N}   : N identical, sweeping N to find saturation (H5)

The PARENT process stays CUDA-free: only the spawned workers touch the GPU, so
the tegrastats-based PowerLogger and MPS control run cleanly in the parent.

Usage on the board:

    ~/xray-venv/bin/python ~/jetson-xray-panel/src/benchmark_concurrent.py \
        --repeats 3 --duration 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import models
import runner_concurrent
import utils
from power_logger import PowerLogger

POWER_MODE = "MAXN_SUPER"
RUNTIME = "pytorch"
RESULTS_DIR = os.path.expanduser("~/jetson-xray-panel/results/raw")
MPS_PIPE = "/tmp/nvidia-mps"
MPS_LOG = "/tmp/nvidia-mps-log"


# --------------------------------------------------------------------------- #
# CUDA MPS control
# --------------------------------------------------------------------------- #

def start_mps() -> None:
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False)
    time.sleep(1.5)
    print("  MPS daemon started")


def stop_mps() -> None:
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n",
                       text=True, timeout=10, check=False)
        print("  MPS daemon stopped")
    except Exception as e:
        print(f"  (MPS stop warning: {e})")


# --------------------------------------------------------------------------- #
# Record assembly / IO
# --------------------------------------------------------------------------- #

def _assemble(config: str, case: str, model_names: list[str], result: dict,
              power: dict, mps: bool, repeat_index: int) -> dict:
    thr = result["n_images"] / result["wall_s"]
    lat = utils.latency_stats(result["per_call_ms"])
    p_mean = power["power_w"]["mean"]
    return {
        "config": config,
        "regime": "concurrent",
        "concurrency_case": case,          # "same" | "diff" | "ramp"
        "mps": mps,
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
        "mem_mb_peak": result["mem_mb_peak_sum"],
        "throughput_per_watt": round(thr / p_mean, 3) if p_mean else None,
        "per_worker": result["per_worker"],
        "repeat_index": repeat_index,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": "Orin",
    }


def _write(rec: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fn = f"{rec['config']}_r{rec['repeat_index']}_{int(time.time() * 1000)}.json"
    with open(os.path.join(RESULTS_DIR, fn), "w") as f:
        json.dump(rec, f, indent=2)


def _measure(config: str, case: str, model_names: list[str], duration: float,
             mps: bool, repeat_index: int, batch_size: int = 1) -> None:
    with PowerLogger(interval_ms=100) as p:
        result = runner_concurrent.run_concurrent(
            model_names, duration=duration, batch_size=batch_size)
    # Align power to just the concurrent window (excludes spawn/load/warmup).
    power = p.summary(result["t_run_start"], result["t_run_end"])
    rec = _assemble(config, case, model_names, result, power, mps, repeat_index)
    _write(rec)
    print(f"  [{config} r{repeat_index}] {rec['throughput_ips']:>7.2f} img/s | "
          f"lat {rec['latency_ms']['mean']:.1f}ms | {rec['power_w']['mean']}W | "
          f"{rec['throughput_per_watt']} img/s/W | GPU {rec['gpu_util_pct']}%")
    if case == "hetero":                      # show the per-model breakdown
        for w in result["per_worker"]:
            print(f"        - {w['model']:26s} {w['throughput_ips']:>6.1f} img/s "
                  f"| {w['latency_ms_mean']:>6.1f} ms/img")


# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #

def build_matrix(same_ns, diff_ns, ramp_ns, batch=1):
    """Return a list of (config, case, model_names). ``batch`` suffixes the config
    name (e.g. C4b2_same) when >1 so concurrent-batched runs stay distinct."""
    workhorse = "densenet121-res224-all"
    bs = "" if batch == 1 else f"b{batch}"
    jobs = []
    for n in same_ns:
        jobs.append((f"C{n}{bs}_same", "same", [workhorse] * n))
    for n in diff_ns:
        jobs.append((f"C{n}{bs}_diff", "diff", models.distinct_panel(n)))
    for n in ramp_ns:
        jobs.append((f"RAMP{n}{bs}", "ramp", [workhorse] * n))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--same", default="2,4,8")
    ap.add_argument("--diff", default="2,4,8")
    ap.add_argument("--ramp", default="1,3,5,6", help="extra ramp points (same-model)")
    ap.add_argument("--batch", type=int, default=1,
                    help="per-model batch size (>1 = concurrent-batched)")
    ap.add_argument("--panel", default=None,
                    help="explicit comma-separated model list -> one 'HETERO' config "
                         "(e.g. a mixed-architecture chest-X-ray panel)")
    args = ap.parse_args()

    mps = not args.no_mps
    if args.panel:
        names = [m.strip() for m in args.panel.split(",") if m.strip()]
        jobs = [("HETERO", "hetero", names)]
    else:
        same_ns = [int(x) for x in args.same.split(",") if x]
        diff_ns = [int(x) for x in args.diff.split(",") if x]
        ramp_ns = [int(x) for x in args.ramp.split(",") if x]
        jobs = build_matrix(same_ns, diff_ns, ramp_ns, batch=args.batch)

    print(f"Concurrent matrix: {[j[0] for j in jobs]}")
    print(f"MPS={mps}, duration={args.duration}s, repeats={args.repeats}, "
          f"batch={args.batch}\n")

    if mps:
        start_mps()
    try:
        for config, case, names in jobs:
            print(f"== {config} ({case}, {len(names)} models x batch {args.batch}) ==")
            for r in range(1, args.repeats + 1):
                _measure(config, case, names, args.duration, mps, r,
                         batch_size=args.batch)
    finally:
        if mps:
            stop_mps()
    print(f"\nDone. Records in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
