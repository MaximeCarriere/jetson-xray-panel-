"""Live multi-clinician demo (Section 9 of PLAN.md).

Simulates several clinicians hitting ONE $249 Jetson at the same time, each asking
a DIFFERENT question -> a DIFFERENT chest-X-ray model -> which cannot be batched
together, so they run concurrently. A live terminal dashboard shows each
clinician's throughput plus the shared box's power and GPU utilisation, proving
they run in parallel on one GPU with no cloud.

Run on the board (best recorded full-screen):

    ~/xray-venv/bin/python ~/jetson-xray-panel/demo/demo_panel.py --seconds 25

The parent process stays CUDA-free; each clinician is its own spawned process
(with CUDA MPS enabled for true concurrent execution — start it first, see
scripts/run_concurrent.sh, or the demo runs without it as time-slicing).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import torch
import torch.multiprocessing as mp

import models
import utils
from power_logger import PowerLogger

_MPS_PIPE, _MPS_LOG = "/tmp/nvidia-mps", "/tmp/nvidia-mps-log"


def _start_mps():
    """Enable CUDA MPS so the clinician processes run truly concurrently."""
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = _MPS_PIPE
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = _MPS_LOG
    os.makedirs(_MPS_PIPE, exist_ok=True)
    os.makedirs(_MPS_LOG, exist_ok=True)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False)
    time.sleep(1.0)


def _stop_mps():
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True,
                   timeout=10, check=False)

# (display name, clinical question, model weights, per-model batch size)
CLINICIANS = [
    ("Dr. A · Screening", "Pneumonia / lung opacity", "densenet121-res224-nih", 1),
    ("Dr. B · Cardiology", "Cardiomegaly / effusion", "densenet121-res224-chex", 1),
    ("ER Dr. C · Panel", "Full multi-pathology read", "densenet121-res224-all", 1),
    ("Radiologist D", "Detailed 512px read", "resnet50-res512-all", 1),
]

# ANSI helpers
CLR = "\x1b[2J\x1b[H"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GRN = "\x1b[32m"
CYN = "\x1b[36m"
YEL = "\x1b[33m"
RST = "\x1b[0m"


def _worker(rank, model_name, batch_size, warmup, start_evt, stop_evt, q):
    """One clinician: load model, warm up, then stream inferences and report."""
    torch.cuda.init()
    utils.seed_everything(rank)
    size = models.input_size_for(model_name)
    model = models.load_model(model_name, device="cuda")
    x = utils.build_input_pool(size=size, n=max(8, batch_size), device="cuda")

    def one():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(x[:batch_size])

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    q.put({"rank": rank, "ready": True})

    start_evt.wait()
    count = 0
    win_c, win_t, last = 0, time.perf_counter(), time.perf_counter()
    while not stop_evt.is_set():
        ts = time.perf_counter()
        one()
        torch.cuda.synchronize()
        lat = (time.perf_counter() - ts) * 1000
        count += batch_size
        win_c += batch_size
        now = time.perf_counter()
        if now - last >= 0.3:
            q.put({"rank": rank, "images": count, "ips": win_c / (now - win_t),
                   "last_ms": lat})
            win_c, win_t, last = 0, now, now


def _bar(pct, width=22):
    if pct is None:
        return " " * width
    fill = int(round((pct / 100) * width))
    return "▓" * fill + "░" * (width - fill)


def _render(elapsed, state, power, n):
    L = []
    L.append(f"{BOLD}{CYN}  ONE $249 JETSON ORIN NANO — LIVE CHEST X-RAY PANEL{RST}"
             f"{DIM}     t = {elapsed:4.1f}s{RST}")
    L.append(f"{DIM}  {n} clinicians · {n} different models · one shared GPU · no cloud{RST}")
    L.append("")
    L.append(f"  {'Clinician':<20}{'Question':<28}{'img/s':>7}{'  last':>8}"
             f"{'  total':>9}")
    L.append(f"  {DIM}{'─' * 70}{RST}")
    agg = 0.0
    for rank, (name, q, model, _) in enumerate(CLINICIANS[:n]):
        s = state.get(rank, {})
        ips = s.get("ips", 0.0)
        agg += ips
        L.append(f"  {name:<20}{q:<28}{GRN}{ips:>6.1f}{RST}"
                 f"{s.get('last_ms', 0):>7.0f}ms{s.get('images', 0):>9}")
    L.append(f"  {DIM}{'─' * 70}{RST}")
    L.append(f"  {BOLD}AGGREGATE — all in parallel{RST}{'':<15}{BOLD}{GRN}{agg:>6.1f} img/s{RST}")
    L.append("")
    gpu = power.get("gpu_util_pct")
    pw = power.get("power_w")
    tmp = power.get("temp_c")
    temp_s = f"{tmp:.0f}" if tmp is not None else "--"
    L.append(f"  GPU {YEL}{_bar(gpu)}{RST} {gpu if gpu is not None else '--':>3}%"
             f"    Power {BOLD}{pw if pw is not None else '--'} W{RST}"
             f"    Temp {temp_s}°C")
    L.append(f"  {DIM}different questions → different models → cannot be batched → "
             f"concurrency is the only way{RST}")
    return CLR + "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--clients", type=int, default=len(CLINICIANS))
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--no-mps", action="store_true", help="disable CUDA MPS")
    args = ap.parse_args()
    n = min(args.clients, len(CLINICIANS))

    if not args.no_mps:
        _start_mps()

    ctx = mp.get_context("spawn")
    start_evt, stop_evt, q = ctx.Event(), ctx.Event(), ctx.Queue()
    procs = []
    state = {}
    try:
        for rank in range(n):
            _, _, model_name, bs = CLINICIANS[rank]
            p = ctx.Process(target=_worker,
                            args=(rank, model_name, bs, args.warmup, start_evt, stop_evt, q))
            p.start()
            procs.append(p)

        # Wait for every clinician to finish loading + warming up.
        print("  Loading models (first run downloads weights)…", flush=True)
        ready = 0
        while ready < n:
            msg = q.get()
            if msg.get("ready"):
                ready += 1
                print(f"  ready {ready}/{n}", flush=True)

        power = {}
        with PowerLogger(interval_ms=200) as plog:
            start_evt.set()
            t0 = time.perf_counter()
            while (elapsed := time.perf_counter() - t0) < args.seconds:
                while not q.empty():             # drain latest per-clinician updates
                    m = q.get()
                    if "rank" in m and "ips" in m:
                        state[m["rank"]] = m
                power = plog.latest()
                sys.stdout.write(_render(elapsed, state, power, n))
                sys.stdout.flush()
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        if not args.no_mps:
            _stop_mps()

    total = sum(state.get(r, {}).get("images", 0) for r in range(n))
    print(f"\n  {BOLD}{GRN}Done.{RST} {n} models served ~{total} images in "
          f"{args.seconds:.0f}s on one $249 box.\n")


if __name__ == "__main__":
    main()
