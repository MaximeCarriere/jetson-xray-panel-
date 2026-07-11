"""TensorRT-powered live multi-clinician demo.

Same multi-clinician story as demo_panel.py, but each clinician runs a **TensorRT
FP16 engine** instead of PyTorch — and all of them run concurrently *in one
process* via CUDA streams (TensorRT engines overlap where PyTorch streams did not,
and share memory with no per-process wall). Live throughput jumps from ~75 img/s
(PyTorch) into the hundreds.

    ~/xray-venv/bin/python ~/jetson-xray-panel/demo/demo_panel_trt.py --seconds 25

Engines default to the ones built by the TensorRT track (nih + all); pass --engines
to use others.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import tensorrt as trt
import torch

from power_logger import PowerLogger

# (display name, clinical question, engine path)
DEFAULT_CLINICIANS = [
    ("Dr. A · Screening", "Pneumonia / lung opacity", "/home/a/densenet_nih_fp16.engine"),
    ("Dr. B · Full panel", "18-pathology read", "/home/a/densenet_all.engine"),
    ("ER Dr. C · Screening", "Effusion / cardiomegaly", "/home/a/densenet_nih_fp16.engine"),
    ("Radiologist D", "18-pathology read", "/home/a/densenet_all.engine"),
]

CLR, BOLD, DIM, GRN, CYN, YEL, RST = ("\x1b[2J\x1b[H", "\x1b[1m", "\x1b[2m",
                                      "\x1b[32m", "\x1b[36m", "\x1b[33m", "\x1b[0m")


class Engine:
    """One TRT engine + context, pre-bound to fixed batch-1 I/O on its own stream."""
    def __init__(self, path, logger):
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        in_name = out_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                in_name = n
            else:
                out_name = n
        n_out = int(self.engine.get_tensor_shape(out_name)[-1])
        self.inp = torch.rand(1, 1, 224, 224, device="cuda")
        self.out = torch.empty(1, n_out, device="cuda")
        self.ctx.set_input_shape(in_name, (1, 1, 224, 224))
        self.ctx.set_tensor_address(in_name, self.inp.data_ptr())
        self.ctx.set_tensor_address(out_name, self.out.data_ptr())
        self.stream = torch.cuda.Stream()

    def issue(self):
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)


def _bar(pct, width=22):
    if pct is None:
        return " " * width
    fill = int(round(pct / 100 * width))
    return "▓" * fill + "░" * (width - fill)


def _render(elapsed, names, questions, ips, counts, power, n):
    L = [f"{BOLD}{CYN}  ONE $249 JETSON — LIVE CHEST X-RAY PANEL  ·  TensorRT FP16{RST}"
         f"{DIM}   t = {elapsed:4.1f}s{RST}",
         f"{DIM}  {n} clinicians · concurrent TensorRT engines · one GPU · no cloud{RST}", "",
         f"  {'Clinician':<22}{'Question':<26}{'img/s':>7}{'  total':>10}",
         f"  {DIM}{'─' * 70}{RST}"]
    for i in range(n):
        L.append(f"  {names[i]:<22}{questions[i]:<26}{GRN}{ips[i]:>6.0f}{RST}{counts[i]:>10}")
    L.append(f"  {DIM}{'─' * 70}{RST}")
    L.append(f"  {BOLD}AGGREGATE — all in parallel{RST}{'':<17}{BOLD}{GRN}{sum(ips):>6.0f} img/s{RST}")
    L.append("")
    gpu, pw, tmp = power.get("gpu_util_pct"), power.get("power_w"), power.get("temp_c")
    temp_s = f"{tmp:.0f}" if tmp is not None else "--"
    L.append(f"  GPU {YEL}{_bar(gpu)}{RST} {gpu if gpu is not None else '--':>3}%"
             f"    Power {BOLD}{pw if pw is not None else '--'} W{RST}    Temp {temp_s}°C")
    L.append(f"  {DIM}TensorRT engines run concurrently in one process — no memory wall{RST}")
    return CLR + "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--engines", nargs="*", default=None)
    args = ap.parse_args()

    clin = DEFAULT_CLINICIANS if not args.engines else [
        (f"Clinician {i+1}", "chest X-ray read", e) for i, e in enumerate(args.engines)]
    n = len(clin)
    names = [c[0] for c in clin]
    questions = [c[1] for c in clin]

    logger = trt.Logger(trt.Logger.ERROR)
    print("  Loading TensorRT engines…", flush=True)
    engs = [Engine(c[2], logger) for c in clin]

    for _ in range(30):                         # warm up
        for e in engs:
            e.issue()
    torch.cuda.synchronize()

    counts = [0] * n
    ips = [0.0] * n
    power = {}
    with PowerLogger(interval_ms=200) as plog:
        t0 = time.perf_counter()
        last, win = t0, [0] * n
        while (elapsed := time.perf_counter() - t0) < args.seconds:
            for e in engs:
                e.issue()
            torch.cuda.synchronize()
            for i in range(n):
                counts[i] += 1
                win[i] += 1
            now = time.perf_counter()
            if now - last >= 0.3:
                ips = [w / (now - last) for w in win]
                win = [0] * n
                last = now
                power = plog.latest()
                sys.stdout.write(_render(elapsed, names, questions, ips, counts, power, n))
                sys.stdout.flush()

    total = sum(counts)
    print(f"\n  {BOLD}{GRN}Done.{RST} {n} TensorRT models served {total} images in "
          f"{args.seconds:.0f}s on one $249 box.\n")


if __name__ == "__main__":
    main()
