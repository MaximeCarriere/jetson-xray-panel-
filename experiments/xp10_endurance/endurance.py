"""Sustained-load thermal endurance test.

Runs a continuous TensorRT FP16 batch-8 workload for N minutes, sampling
throughput, temperature, power, and GPU utilisation every 10 s. Answers: does the
box hold its throughput under sustained load, or does it thermally throttle? —
i.e. is it viable as an always-on clinic device, not just a benchmark burst.

    ~/xray-venv/bin/python endurance.py ~/densenet_nih_fp16.engine --minutes 20
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
from power_logger import PowerLogger
from trt_runner import TRTModel

RESULTS = "/home/a/jetson-xray-panel/results/endurance.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", nargs="?", default="/home/a/densenet_nih_fp16.engine")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    model = TRTModel(args.engine)
    x = torch.rand(args.batch, 1, 224, 224, device="cuda")
    for _ in range(30):
        model.infer(x)
    torch.cuda.synchronize()

    samples = []
    with PowerLogger(interval_ms=200) as plog:
        t0 = time.perf_counter()
        last, win_n = t0, 0
        while (elapsed := time.perf_counter() - t0) < args.minutes * 60:
            model.infer(x)
            win_n += args.batch
            now = time.perf_counter()
            if now - last >= args.interval:
                lat = plog.latest()
                s = {"t_s": round(now - t0, 1),
                     "ips": round(win_n / (now - last), 1),
                     "power_w": lat["power_w"], "temp_c": lat["temp_c"],
                     "gpu_util_pct": lat["gpu_util_pct"]}
                samples.append(s)
                print(f"  t={s['t_s']:6.0f}s  {s['ips']:6.1f} img/s  "
                      f"{s['temp_c']}C  {s['power_w']}W  GPU {s['gpu_util_pct']}%",
                      flush=True)
                win_n, last = 0, now

    ips = [s["ips"] for s in samples]
    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    first, last_avg = ips[0], sum(ips[-3:]) / len(ips[-3:])
    drop_pct = (first - last_avg) / first * 100
    summary = {
        "minutes": args.minutes, "batch": args.batch,
        "throughput_start_ips": first, "throughput_end_ips": round(last_avg, 1),
        "throughput_drop_pct": round(drop_pct, 1),
        "temp_c_max": max(temps) if temps else None,
        "samples": samples,
    }
    with open(RESULTS, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== endurance: start {first:.0f} -> end {last_avg:.0f} img/s "
          f"({drop_pct:+.1f}%), peak {summary['temp_c_max']}C ===")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
