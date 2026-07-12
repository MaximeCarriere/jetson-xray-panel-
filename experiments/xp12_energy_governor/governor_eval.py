"""XP12 — energy vs SLA under a realistic clinic-day load, three power policies.

Replays a bursty load profile (quiet periods + rounds/peaks) through the XP11
serving layer under three policies — always-MAXN, always-25W, and the adaptive
governor — and reports energy (Joules/image), average power, and p99 latency +
SLA-violation rate for each. The point: the governor should hold the SLA like
MAXN while using far less energy, because most of a real day is not peak load.

    ~/xray-venv/bin/python governor_eval.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "lib"))
sys.path.insert(0, os.path.join(_HERE, "..", "xp11_serving"))

import numpy as np

from governor import Governor, MODES, set_mode
from power_logger import PowerLogger
from serving import Server

# clinic-day profile: (duration_s, target_req/s) — quiet ↔ rounds ↔ peak.
# Peaks (440/450) sit above 25W capacity (~430) but below MAXN (~482), so 25W fails
# the SLA while MAXN and the adaptive governor can hold it.
PROFILE = [(12, 60), (12, 440), (14, 120), (10, 450), (12, 70), (12, 290)]
SLA_MS = 100.0
ENGINE = "/home/a/densenet_nih_fp16.engine"
RESULTS = "/home/a/jetson-xray-panel/results/governor_bench.json"


def feed_profile(server, sink):
    rng = random.Random(0)
    for dur, rps in PROFILE:
        seg_end = time.perf_counter() + dur
        nxt = time.perf_counter()
        while time.perf_counter() < seg_end:
            now = time.perf_counter()
            if now >= nxt:
                server.submit("nih", {"t_submit": now, "sink": sink})
                nxt += rng.expovariate(rps)
            else:
                time.sleep(min(0.0003, max(0.0, nxt - now)))


def run_policy(name, adaptive, fixed_mode):
    server = Server([("nih", ENGINE)], max_batch=8, max_delay_ms=5.0)
    server.start()
    gov = Governor(server) if adaptive else None
    sink = []
    with PowerLogger(interval_ms=200) as plog:
        if adaptive:
            gov.start()
        else:
            set_mode(fixed_mode)
        time.sleep(2.0)                         # let the mode settle before timing
        t0 = time.perf_counter()
        feed_profile(server, sink)
        server.drain()
        t1 = time.perf_counter()
        if adaptive:
            gov.stop()
            time.sleep(0.3)
    server.stop()

    lat = np.array([l for (_ts, l) in sink], dtype=float)
    energy = plog.energy_joules(t0, t1)
    n = len(lat)
    res = {
        "policy": name,
        "energy_j": round(energy, 1),
        "j_per_img": round(energy / n, 4),
        "avg_power_w": round(energy / (t1 - t0), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
        "sla_violation_pct": round(100 * float(np.mean(lat > SLA_MS)), 1),
        "n": n,
    }
    if adaptive:
        res["mode_timeline"] = gov.timeline
    return res


def main():
    results = []
    try:
        for name, adaptive, mode in [("always-MAXN", False, 2),
                                     ("always-25W", False, 1),
                                     ("adaptive", True, None)]:
            print(f">>> {name}", flush=True)
            r = run_policy(name, adaptive, mode)
            results.append(r)
            print(f"  {r['energy_j']} J ({r['j_per_img']} J/img, {r['avg_power_w']} W avg) "
                  f"| p99 {r['p99_ms']} ms | SLA miss {r['sla_violation_pct']}%", flush=True)
    finally:
        set_mode(2)                             # restore MAXN_SUPER

    base = next((r for r in results if r["policy"] == "always-MAXN"), None)
    if base:
        for r in results:
            r["energy_vs_maxn_pct"] = round(100 * (r["energy_j"] / base["energy_j"] - 1), 1)

    with open(RESULTS, "w") as f:
        json.dump({"profile": PROFILE, "sla_ms": SLA_MS, "results": results}, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
