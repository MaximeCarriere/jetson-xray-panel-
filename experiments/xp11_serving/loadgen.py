"""XP11 — open-loop Poisson load generator for the serving layer.

Fires requests at a target rate (exponential inter-arrival = Poisson process),
fire-and-forget, so slow responses never throttle new arrivals (open-loop, the
right model for a latency study). Sweeps the target RPS and reports the end-to-end
latency distribution (p50/p95/p99) and achieved throughput at each level, so we can
draw the latency-vs-load curve and read off the SLA-bounded capacity.

    ~/xray-venv/bin/python loadgen.py --rps 50,100,200,300,400,500,600,700 \
        --duration 8 --sla-p99-ms 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

import numpy as np

from serving import Server

RESULTS = "/home/a/jetson-xray-panel/results/serving_bench.json"


def run_rps(server, rps, duration, models, warmup=1.5):
    sink = []
    rng = random.Random(0)
    t_start = time.perf_counter()
    end = t_start + warmup + duration
    next_t = t_start
    while time.perf_counter() < end:
        now = time.perf_counter()
        if now >= next_t:
            m = rng.choice(models)
            server.submit(m, {"t_submit": now, "sink": sink})
            next_t += rng.expovariate(rps)
        else:
            time.sleep(min(0.0003, max(0.0, next_t - now)))
    server.drain()

    meas_start = t_start + warmup
    lat = np.array([l for (ts, l) in sink if ts >= meas_start], dtype=float)
    if len(lat) == 0:
        return None
    return {
        "rps_target": rps,
        "rps_achieved": round(len(lat) / duration, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "mean_ms": round(float(lat.mean()), 2),
        "n": int(len(lat)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", default="50,100,200,300,400,500,600,700")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-delay-ms", type=float, default=5.0)
    ap.add_argument("--sla-p99-ms", type=float, default=100.0)
    ap.add_argument("--engine", default="/home/a/densenet_nih_fp16.engine")
    args = ap.parse_args()

    specs = [("nih", args.engine)]
    server = Server(specs, max_batch=args.max_batch, max_delay_ms=args.max_delay_ms)
    server.start()
    print(f"serving: max_batch={args.max_batch}, max_delay={args.max_delay_ms}ms, "
          f"SLA p99<{args.sla_p99_ms}ms\n")
    print(f"{'target':>7} {'achieved':>9} {'p50':>7} {'p95':>7} {'p99':>7} {'SLA':>5}")

    rps_list = [int(x) for x in args.rps.split(",")]
    rows = []
    sla_cap = 0
    for rps in rps_list:
        r = run_rps(server, rps, args.duration, [s[0] for s in specs])
        if r is None:
            continue
        ok = r["p99_ms"] <= args.sla_p99_ms
        r["sla_ok"] = ok
        if ok:
            sla_cap = max(sla_cap, r["rps_achieved"])
        rows.append(r)
        print(f"{rps:>7} {r['rps_achieved']:>9.1f} {r['p50_ms']:>7.1f} "
              f"{r['p95_ms']:>7.1f} {r['p99_ms']:>7.1f} {'ok' if ok else 'MISS':>5}")

    server.stop()
    out = {"max_batch": args.max_batch, "max_delay_ms": args.max_delay_ms,
           "sla_p99_ms": args.sla_p99_ms, "sla_bounded_rps": sla_cap,
           "batch_stats": server.batch_stats(), "rows": rows}
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSLA-bounded capacity: ~{sla_cap:.0f} req/s (p99 < {args.sla_p99_ms:.0f} ms)")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
