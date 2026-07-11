"""Export a torchxrayvision model to ONNX for TensorRT engine building.

TensorRT gives edge-deployment speed via layer fusion + kernel auto-tuning for
this exact chip (in FP16 — same precision we already use, so accuracy is
unchanged; we do NOT use INT8 quantization here).

    ~/xray-venv/bin/python trt_export.py densenet121-res224-all ~/densenet_all.onnx --batch 1

Then build an engine with trtexec:
    /usr/src/tensorrt/bin/trtexec --onnx=~/densenet_all.onnx --fp16 \
        --saveEngine=~/densenet_all.engine
"""
from __future__ import annotations

import argparse

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "lib"))
import models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    size = models.input_size_for(args.weights)
    model = models.load_model(args.weights, device="cuda").eval()
    # Drop op_norm: it's a trivial elementwise post-normalization whose traced
    # reshape is batch-1-only and breaks dynamic-batch TensorRT. Setting
    # op_threshs=None makes forward return the raw sigmoid probabilities, which
    # batch cleanly; op_norm (if wanted) is a cheap numpy step applied after.
    model.op_threshs = None
    dummy = torch.randn(args.batch, 1, size, size, device="cuda")

    # Dynamic batch so one engine serves batch 1..N. Use the legacy TorchScript
    # exporter (dynamo=False): torchxrayvision's forward has a data-dependent
    # input-range warning check that the new dynamo exporter can't trace.
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["pathologies"],
        dynamic_axes={"image": {0: "batch"}, "pathologies": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {args.weights} -> {args.out} (input {args.batch}x1x{size}x{size})")


if __name__ == "__main__":
    main()
