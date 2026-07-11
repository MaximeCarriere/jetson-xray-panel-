"""Throughput & efficiency across the board's power modes (15W / 25W / MAXN_SUPER).

Runs one fixed workload (TensorRT FP16, batch 8) at each nvpmodel power mode and
records throughput, power, and throughput-per-watt — to find the most efficient
operating point for a clinic / battery deployment. Restores MAXN_SUPER at the end.

    ~/xray-venv/bin/python power_sweep.py ~/densenet_nih_fp16.engine
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
from power_logger import PowerLogger
from trt_runner import TRTModel

MODES = [(2, "MAXN_SUPER"), (1, "25W"), (0, "15W")]
RESULTS = "/home/a/jetson-xray-panel/results/power_sweep.json"


def _set_mode(m: int) -> None:
    subprocess.run(f"echo a | sudo -S nvpmodel -m {m}", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("echo a | sudo -S jetson_clocks", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)          # let clocks/thermals settle


def _bench(model: TRTModel, batch=8, duration=6.0, warmup=20):
    x = torch.rand(batch, 1, 224, 224, device="cuda")
    for _ in range(warmup):
        model.infer(x)
    torch.cuda.synchronize()
    with PowerLogger(interval_ms=100) as p:
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < duration:
            model.infer(x)
            n += 1
        dt = time.perf_counter() - t0
        pw = p.summary(t0, t0 + dt)
    ips = n * batch / dt
    return ips, pw


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "/home/a/densenet_nih_fp16.engine"
    from stats import agg

    REPEATS = 3
    model = TRTModel(engine)                 # loaded once; reused across modes
    rows = []
    try:
        for m, name in MODES:
            _set_mode(m)
            ips_s, pw_s, eff_s, temp_s = [], [], [], []
            for _ in range(REPEATS):
                ips, pw = _bench(model)
                watt = pw["power_w"]["mean"]
                ips_s.append(ips); pw_s.append(watt); eff_s.append(ips / watt)
                temp_s.append(pw["temp_c_peak"])
            ai, ap, ae = agg(ips_s), agg(pw_s), agg(eff_s)
            row = {"mode": name, "nvpmodel": m,
                   "throughput_ips": round(ai["mean"], 1), "throughput_se": round(ai["se"], 2),
                   "power_w": round(ap["mean"], 2),
                   "throughput_per_watt": round(ae["mean"], 2), "eff_se": round(ae["se"], 3),
                   "temp_c_peak": max(temp_s)}
            rows.append(row)
            print(f"  {name:12s} {ai['mean']:7.1f}±{ai['se']:.1f} img/s | {ap['mean']:5.1f} W | "
                  f"{ae['mean']:6.2f}±{ae['se']:.2f} img/s/W | {max(temp_s):.0f}C")
    finally:
        _set_mode(2)                          # always restore MAXN_SUPER
        print("  restored MAXN_SUPER")
    with open(RESULTS, "w") as f:
        json.dump({"engine": engine.split("/")[-1], "batch": 8, "repeats": REPEATS,
                   "rows": rows}, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
