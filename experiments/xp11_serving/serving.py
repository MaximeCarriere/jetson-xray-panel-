"""XP11 — dynamic-batching inference server (serving layer).

A lightweight, dependency-free serving core: each model has a request queue and a
dedicated worker thread that owns its TensorRT execution context and CUDA stream.
The worker forms a batch when it has ``max_batch`` requests OR ``max_delay_ms`` has
elapsed (classic dynamic batching), runs the engine, and completes all the requests
in that batch together. Different models run on separate streams, so the GPU
overlaps them (as measured in XP6).

This is the substrate for the load test (loadgen.py) and the energy governor (XP12).
Requests carry a ``t_submit`` timestamp; end-to-end latency = queue wait + batch
formation delay + inference. Payload transfer is out of scope here — we submit a
model id and the worker runs on a resident tensor, to isolate serving+compute
latency (content does not affect timing).
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

import torch

from trt_runner import TRTModel


class DynamicBatcher:
    def __init__(self, name, engine_path, max_batch=8, max_delay_ms=5.0, size=224):
        self.name = name
        self.engine_path = engine_path
        self.max_batch = max_batch
        self.max_delay = max_delay_ms / 1000.0
        self.size = size
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.batch_sizes = []                 # every batch formed (for utilisation stats)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        self.ready.wait()

    def submit(self, job: dict):
        """job: {'t_submit': float, optional 'sink': list, optional 'event': Event}."""
        self.q.put(job)

    def _run(self):
        model = TRTModel(self.engine_path)     # context owned by this worker thread
        stream = torch.cuda.Stream()           # its own stream -> GPU overlaps models
        x1 = torch.rand(1, 1, self.size, self.size, device="cuda")
        with torch.cuda.stream(stream):
            for _ in range(15):
                model.infer(x1)
        torch.cuda.synchronize()
        self.ready.set()

        while not self.stop.is_set():
            try:
                jobs = [self.q.get(timeout=0.2)]
            except queue.Empty:
                continue
            deadline = time.perf_counter() + self.max_delay
            while len(jobs) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    jobs.append(self.q.get(timeout=remaining))
                except queue.Empty:
                    break

            b = len(jobs)
            self.batch_sizes.append(b)
            x = x1.expand(b, -1, -1, -1).contiguous()
            with torch.cuda.stream(stream):
                model.infer(x)                 # TRTModel.infer syncs this stream
            done = time.perf_counter()
            for job in jobs:
                job["t_done"] = done
                if "sink" in job:
                    # (submit_time, end_to_end_latency_ms) — submit_time lets the
                    # load generator drop warm-up requests.
                    job["sink"].append((job["t_submit"], (done - job["t_submit"]) * 1000.0))
                if "event" in job:
                    job["event"].set()


class Server:
    def __init__(self, specs, max_batch=8, max_delay_ms=5.0):
        """specs: list of (name, engine_path)."""
        self.batchers = {
            name: DynamicBatcher(name, path, max_batch, max_delay_ms)
            for name, path in specs
        }
        self.submitted = 0                     # total requests (governor reads the rate)

    def start(self):
        for b in self.batchers.values():
            b.start()

    def submit(self, model: str, job: dict):
        self.submitted += 1                    # GIL-atomic; a live-load signal
        self.batchers[model].submit(job)

    def drain(self, timeout=15.0):
        t0 = time.perf_counter()
        while any(not b.q.empty() for b in self.batchers.values()):
            if time.perf_counter() - t0 > timeout:
                break
            time.sleep(0.01)
        time.sleep(0.05)                       # let the last in-flight batch complete

    def stop(self):
        for b in self.batchers.values():
            b.stop.set()

    def batch_stats(self):
        out = {}
        for name, b in self.batchers.items():
            bs = b.batch_sizes
            out[name] = {"batches": len(bs),
                         "mean_batch": round(sum(bs) / len(bs), 2) if bs else 0}
        return out
