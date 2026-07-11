"""Build an INT8 TensorRT engine with calibration on real chest X-rays.

INT8 quantization (unlike the FP16 used elsewhere) *can* shift predictions, so it
must be calibrated on representative data and its accuracy verified. We calibrate
on ChestMNIST images and later measure AUROC (trt_eval_auroc.py) to give an honest
speed-vs-accuracy verdict.

Uses a torch CUDA tensor as the calibrator's device buffer (no pycuda).

    ~/xray-venv/bin/python trt_int8.py ~/densenet_nih.onnx ~/densenet_nih_int8.engine
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorrt as trt
import torch


def _load_calib(n=512, size=224):
    """ChestMNIST images -> xrv-normalized (n,1,H,W) float32. Uses a slice disjoint
    from the AUROC eval set (eval uses test[:2000])."""
    from medmnist import ChestMNIST
    ds = ChestMNIST(split="test", size=size, download=True)
    imgs = ds.imgs[2000:2000 + n].astype(np.float32)          # (n,H,W) uint8->float
    x = (2.0 * (imgs / 255.0) - 1.0) * 1024.0                  # xrv range
    return x[:, None, :, :].copy()                            # (n,1,H,W)


class ChestCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data, batch, cache_path):
        super().__init__()
        self.data = data
        self.batch = batch
        self.idx = 0
        self.cache_path = cache_path
        self.dev = torch.empty((batch, 1, data.shape[2], data.shape[3]),
                               dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.idx + self.batch > len(self.data):
            return None
        chunk = self.data[self.idx:self.idx + self.batch]
        self.dev.copy_(torch.from_numpy(chunk))
        self.idx += self.batch
        return [int(self.dev.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def build_int8(onnx_path, engine_path, opt_batch=8, max_batch=16):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)          # explicit batch (TRT 10 default)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)        # fp16 fallback for int8-unfriendly layers

    calib = ChestCalibrator(_load_calib(), opt_batch, engine_path + ".calib")
    config.int8_calibrator = calib

    profile = builder.create_optimization_profile()
    profile.set_shape("image", (1, 1, 224, 224),
                      (opt_batch, 1, 224, 224), (max_batch, 1, 224, 224))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    print("Building INT8 engine (calibrating on 512 ChestMNIST images)…", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("INT8 engine build failed")
    blob = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"wrote {engine_path} ({len(blob)/1e6:.1f} MB)")


if __name__ == "__main__":
    onnx = sys.argv[1] if len(sys.argv) > 1 else "/home/a/densenet_nih.onnx"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/a/densenet_nih_int8.engine"
    build_int8(onnx, out)
