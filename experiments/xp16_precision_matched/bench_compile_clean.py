"""Self-consistent compilation benchmark: throughput AND accuracy, one model.

Everything below uses densenet121-res224-nih, batch 8, op_threshs=None, and the
same 2000 ChestMNIST-224 test images. That makes the blog post's two claims
(3x faster, accuracy unchanged) describe the SAME model, which the previous
numbers did not: throughput came from the "-all" weights and AUROC from "-nih".

Rows:
  torch_fp32           PyTorch, full FP32
  torch_fp16_autocast  PyTorch, torch.autocast(float16)
  torch_fp16_half      PyTorch, model.half(), true FP16 weights
  trt_fp16             TensorRT FP16 engine

Headline = trt_fp16 vs the FASTEST PyTorch FP16 row. Same precision, same
weights, same batch: the only difference left is compilation.
"""
import json
import statistics
import sys
import time

sys.path.insert(0, "/home/a/jetson-xray-panel")
sys.path.insert(0, "/home/a/jetson-xray-panel/lib")

import numpy as np
import torch

import chest_labels as cl
import models
import utils
from stats import bootstrap_auroc
from trt_runner import TRTModel

BATCH = 8
N_BATCHES = 50
REPEATS = 3
WARMUP = 10
N_EVAL = 2000
WEIGHTS = "densenet121-res224-nih"
ENGINE = "/home/a/densenet_nih_fp16.engine"


def time_steps(step_fn):
    for _ in range(WARMUP):
        step_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        step_fn()
    torch.cuda.synchronize()
    return (N_BATCHES * BATCH) / (time.perf_counter() - t0)


def torch_step(model, x, autocast):
    def step():
        with torch.no_grad():
            if autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
    return step


def load(half=False):
    m = models.load_model(WEIGHTS, device="cuda").eval()
    m.op_threshs = None
    return m.half() if half else m


def auroc_from_preds(labels, preds, tag):
    auc, k = cl.macro_auroc(labels, preds)
    bs = bootstrap_auroc(labels, preds, cl.macro_auroc)
    print(f"  {tag:<22} macro-AUROC {auc:.4f} +/- {bs['se']:.4f}  ({k} labels, {len(labels)} imgs)")
    return {"auroc": round(auc, 4), "se": round(bs["se"], 4)}


def main():
    utils.seed_everything(0)
    print(f"device: {torch.cuda.get_device_name(0)} | model {WEIGHTS}")
    print(f"batch {BATCH}, {N_BATCHES} batches/run, {REPEATS} repeats\n")

    pool = utils.build_input_pool(224, BATCH, device="cuda").contiguous()
    x32 = pool[:BATCH].contiguous()

    # ---------------- throughput ----------------
    thr = {}
    m = load()
    thr["torch_fp32"] = [time_steps(torch_step(m, x32, False)) for _ in range(REPEATS)]
    thr["torch_fp16_autocast"] = [time_steps(torch_step(m, x32, True)) for _ in range(REPEATS)]
    del m
    torch.cuda.empty_cache()

    mh = load(half=True)
    x16 = x32.half().contiguous()
    thr["torch_fp16_half"] = [time_steps(torch_step(mh, x16, False)) for _ in range(REPEATS)]
    del mh
    torch.cuda.empty_cache()

    trt = TRTModel(ENGINE)
    thr["trt_fp16"] = [time_steps(lambda: trt.infer(x32)) for _ in range(REPEATS)]

    best_torch_fp16 = max(statistics.mean(thr["torch_fp16_autocast"]),
                          statistics.mean(thr["torch_fp16_half"]))
    print(f"{'config':<22} {'img/s (mean +/- SE)':<24} {'vs best FP16 PyTorch':>21}")
    print("-" * 68)
    summary = {}
    for name, vals in thr.items():
        mm = statistics.mean(vals)
        se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
        summary[name] = {"mean_ips": round(mm, 1), "se_ips": round(se, 1),
                         "runs": [round(v, 1) for v in vals]}
        print(f"{name:<22} {mm:>8.1f} +/- {se:<11.1f} {mm / best_torch_fp16:>20.2f}x")

    speedup = statistics.mean(thr["trt_fp16"]) / best_torch_fp16
    print(f"\ncompilation speedup, precision matched at FP16: {speedup:.2f}x\n")

    # ---------------- accuracy ----------------
    print("accuracy on 2000 ChestMNIST-224 test images:")
    from medmnist import ChestMNIST
    ds = ChestMNIST(split="test", size=224, download=True)
    imgs = ds.imgs[:N_EVAL].astype(np.float32)
    labels = ds.labels[:N_EVAL].astype(np.int64)
    x = torch.from_numpy((2.0 * (imgs / 255.0) - 1.0) * 1024.0)[:, None].cuda()

    cmap = cl.col_map(models.load_model(WEIGHTS, "cuda"))
    acc = {}

    def collect(infer_fn):
        preds = np.full((N_EVAL, len(cl.MEDMNIST_LABELS)), np.nan)
        for i in range(0, N_EVAL, BATCH):
            probs = torch.sigmoid(infer_fn(x[i:i + BATCH].contiguous())).float().cpu().numpy()
            for mc, dc in cmap:
                preds[i:i + BATCH, dc] = probs[:, mc]
        return preds

    mh = load(half=True)
    with torch.no_grad():
        acc["torch_fp16_half"] = auroc_from_preds(
            labels, collect(lambda b: mh(b.half())), "PyTorch FP16")
    del mh
    torch.cuda.empty_cache()

    acc["trt_fp16"] = auroc_from_preds(
        labels, collect(lambda b: trt.infer(b)), "TensorRT FP16")

    out = {"batch_size": BATCH, "n_batches": N_BATCHES, "repeats": REPEATS,
           "n_eval_images": N_EVAL, "weights": WEIGHTS, "engine": ENGINE,
           "device": torch.cuda.get_device_name(0),
           "throughput": summary, "accuracy": acc,
           "compilation_speedup_fp16_matched": round(speedup, 2)}
    path = "/home/a/jetson-xray-panel/results/compile_clean_bench.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
