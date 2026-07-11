"""Shared benchmark utilities: preprocessing, input pools, timing, latency stats.

Measurement hygiene lives here so every runner uses the same warm-up and timing
logic (Section 5 of PLAN.md):
  * warm-up iterations are discarded (CUDA kernel autotuning skews the first runs),
  * throughput is measured from wall-clock over the whole steady-state loop,
  * per-inference latency percentiles come from CUDA events around each call.
"""
from __future__ import annotations

import glob
import os
import random
import time

import numpy as np
import torch

# torchxrayvision preprocessing helpers (center-crop + resize + normalization).
import torchxrayvision as xrv
import skimage.io


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Input pool
# --------------------------------------------------------------------------- #
# NOTE: throughput / latency / power are *content-agnostic* — only the tensor
# shape and dtype affect the GPU work. So for pure systems benchmarking we can
# use either real X-rays (if a dataset is present) or synthetic images of the
# correct shape. Real images only matter for the stretch accuracy/TTA experiment.

def _preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Grayscale -> xrv-normalized, center-cropped, resized (1, size, size)."""
    img = xrv.datasets.normalize(img, 255)          # to [-1024, 1024]
    if img.ndim == 3:                                # RGB -> single channel
        img = img.mean(2)
    img = img[None, ...]                             # add channel dim -> (1, H, W)
    img = xrv.datasets.XRayCenterCrop()(img)         # square center crop
    img = xrv.datasets.XRayResizer(size, engine="cv2")(img)  # -> (1, size, size)
    return img.astype(np.float32)


def load_xray(path: str, size: int = 224) -> torch.Tensor:
    """Load and preprocess one X-ray file to a (1, size, size) float32 tensor."""
    return torch.from_numpy(_preprocess(skimage.io.imread(path), size))


def build_input_pool(size: int, n: int, image_dir: str | None = None,
                     device: str = "cuda") -> torch.Tensor:
    """Return an (n, 1, size, size) FP32 tensor pool, resident on ``device``.

    If ``image_dir`` holds images, cycle through real X-rays; otherwise synthesize
    noise in the xrv value range (content does not affect timing — see note above).
    """
    if image_dir and os.path.isdir(image_dir):
        paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
        )
    else:
        paths = []

    if paths:
        imgs = [_preprocess(skimage.io.imread(paths[i % len(paths)]), size)
                for i in range(n)]
        pool = torch.from_numpy(np.stack(imgs))
    else:
        # Synthetic pool in the xrv range [-1024, 1024]; shape is what matters.
        g = torch.Generator().manual_seed(0)
        pool = (torch.rand(n, 1, size, size, generator=g) * 2048.0) - 1024.0

    return pool.to(device)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def latency_stats(latencies_ms: list[float]) -> dict:
    """mean / p50 / p95 over a list of per-inference latencies (milliseconds)."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def timed_inference(step_fn, n_iter: int, warmup: int = 10):
    """Run ``step_fn`` ``n_iter`` times after ``warmup`` discarded iterations.

    ``step_fn()`` must perform exactly one unit of work (one image, or one batch).
    Returns ``(wall_seconds, per_call_latencies_ms)``:
      * wall_seconds  — total steady-state wall time (throughput = units / wall),
      * per_call_ms   — CUDA-event latency of each call (for percentiles).
    """
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]

    t0 = time.perf_counter()
    for i in range(n_iter):
        starts[i].record()
        step_fn()
        ends[i].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    per_call_ms = [starts[i].elapsed_time(ends[i]) for i in range(n_iter)]
    return wall, per_call_ms


def peak_mem_mb() -> float:
    """Peak CUDA memory allocated since the last reset, in MB."""
    return torch.cuda.max_memory_allocated() / 1e6
