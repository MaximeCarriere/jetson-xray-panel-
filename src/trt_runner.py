"""Minimal TensorRT runtime for the exported chest-X-ray engines.

Runs a TensorRT engine directly on torch CUDA tensors (zero-copy via data_ptr,
no pycuda). Used for the accuracy sanity check (TRT FP16 vs PyTorch) and for the
concurrent-TRT benchmark.

    ~/xray-venv/bin/python trt_runner.py ~/densenet_all.engine   # accuracy + timing
"""
from __future__ import annotations

import sys
import time

import tensorrt as trt
import torch


class TRTModel:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        self.n_out = int(self.engine.get_tensor_shape(self.out_name)[-1])

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) float32 CUDA contiguous -> (B, n_out) float32 CUDA."""
        b = x.shape[0]
        self.context.set_input_shape(self.in_name, tuple(x.shape))
        out = torch.empty((b, self.n_out), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, out.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out


def _accuracy_and_timing(engine_path: str):
    import models
    import utils

    # PyTorch reference (op_norm off to match the exported engine).
    ref_model = models.load_model("densenet121-res224-all", "cuda").eval()
    ref_model.op_threshs = None
    x = utils.build_input_pool(224, 1, device="cuda").contiguous()

    with torch.no_grad():
        ref_fp32 = ref_model(x).float()
    trt_model = TRTModel(engine_path)
    trt_out = trt_model.infer(x)

    # op_threshs=None returns raw logits; sigmoid -> the pathology probabilities a
    # clinician actually reads. Compare those.
    ref_p = torch.sigmoid(ref_fp32)
    trt_p = torch.sigmoid(trt_out)
    diff = (ref_p - trt_p).abs()
    print("=== accuracy: PyTorch FP32 vs TensorRT FP16 (18 pathology probabilities) ===")
    print(f"  max abs diff : {diff.max().item():.5f}")
    print(f"  mean abs diff: {diff.mean().item():.5f}")
    print(f"  probs agree to within {diff.max().item()*100:.2f} percentage points")

    # Single-stream timing (batch 1), warmed up.
    for _ in range(20):
        trt_model.infer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 300
    for _ in range(N):
        trt_model.infer(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"=== single-stream TRT: {N/dt:.1f} img/s ({dt/N*1000:.2f} ms/img) ===")


def _concurrent_streams(engine_path: str, ks=(1, 2, 4, 6, 8), batch=1, duration=4.0):
    """Run K TRT execution contexts, each on its own CUDA stream, and measure
    aggregate throughput. Tests whether TRT engines overlap in one process (where
    PyTorch CUDA streams did not — TRT issues one execute call, not 200 kernels)."""
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_name = n
        else:
            out_name = n
    n_out = int(engine.get_tensor_shape(out_name)[-1])
    size = 224

    print(f"=== concurrent TRT (in-process, one stream per engine, batch {batch}) ===")
    for k in ks:
        ctxs = [engine.create_execution_context() for _ in range(k)]
        ins = [torch.rand(batch, 1, size, size, device="cuda") for _ in range(k)]
        outs = [torch.empty(batch, n_out, device="cuda") for _ in range(k)]
        streams = [torch.cuda.Stream() for _ in range(k)]
        for i in range(k):
            ctxs[i].set_input_shape(in_name, (batch, 1, size, size))
            ctxs[i].set_tensor_address(in_name, ins[i].data_ptr())
            ctxs[i].set_tensor_address(out_name, outs[i].data_ptr())

        def one_round():
            for i in range(k):
                with torch.cuda.stream(streams[i]):
                    ctxs[i].execute_async_v3(streams[i].cuda_stream)
            torch.cuda.synchronize()

        for _ in range(20):
            one_round()
        torch.cuda.synchronize()
        t0, rounds = time.perf_counter(), 0
        while time.perf_counter() - t0 < duration:
            one_round()
            rounds += 1
        dt = time.perf_counter() - t0
        ips = rounds * k * batch / dt
        print(f"  K={k}: {ips:7.1f} img/s   ({ips / k:.1f}/engine)")


if __name__ == "__main__":
    engine = sys.argv[1] if len(sys.argv) > 1 else "/home/a/densenet_all.engine"
    _accuracy_and_timing(engine)
    _concurrent_streams(engine)
