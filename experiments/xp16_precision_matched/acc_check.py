"""Accuracy at matched FP16, plus proof that model.half() is numerically broken here.

torchxrayvision scales inputs to +/-1024. In pure FP16 the early convolutions
overflow, so model.half() is fast but returns garbage. torch.autocast keeps
params FP32 and accumulates in FP32, which is the valid way to run this model
at FP16 in PyTorch. This script shows both.
"""
import sys

sys.path.insert(0, "/home/a/jetson-xray-panel")
sys.path.insert(0, "/home/a/jetson-xray-panel/lib")

import numpy as np
import torch

import chest_labels as cl
import models
from stats import bootstrap_auroc
from trt_runner import TRTModel

N_EVAL = 2000
BATCH = 8
WEIGHTS = "densenet121-res224-nih"
ENGINE = "/home/a/densenet_nih_fp16.engine"


def load(half=False):
    m = models.load_model(WEIGHTS, device="cuda").eval()
    m.op_threshs = None
    return m.half() if half else m


def main():
    from medmnist import ChestMNIST
    ds = ChestMNIST(split="test", size=224, download=True)
    imgs = ds.imgs[:N_EVAL].astype(np.float32)
    labels = ds.labels[:N_EVAL].astype(np.int64)
    x = torch.from_numpy((2.0 * (imgs / 255.0) - 1.0) * 1024.0)[:, None].cuda()
    cmap = cl.col_map(models.load_model(WEIGHTS, "cuda"))

    def collect(fn):
        preds = np.full((N_EVAL, len(cl.MEDMNIST_LABELS)), np.nan)
        bad = 0
        for i in range(0, N_EVAL, BATCH):
            out = fn(x[i:i + BATCH].contiguous())
            bad += int((~torch.isfinite(out)).sum().item())
            probs = torch.sigmoid(out).float().cpu().numpy()
            for mc, dc in cmap:
                preds[i:i + BATCH, dc] = probs[:, mc]
        return preds, bad

    def report(tag, preds, bad):
        auc, k = cl.macro_auroc(labels, preds)
        se = bootstrap_auroc(labels, preds, cl.macro_auroc)["se"]
        flag = f"  [{bad} non-finite outputs]" if bad else ""
        print(f"  {tag:<26} macro-AUROC {auc:.4f} +/- {se:.4f}{flag}")
        return auc

    print(f"model {WEIGHTS}, {N_EVAL} ChestMNIST-224 test images\n")

    m = load()
    with torch.no_grad():
        def ac(b):
            with torch.autocast("cuda", dtype=torch.float16):
                return m(b).float()
        report("PyTorch FP16 (autocast)", *collect(ac))
        report("PyTorch FP32", *collect(lambda b: m(b)))
    del m
    torch.cuda.empty_cache()

    mh = load(half=True)
    with torch.no_grad():
        report("PyTorch FP16 (model.half)", *collect(lambda b: mh(b.half()).float()))
    del mh
    torch.cuda.empty_cache()

    trt = TRTModel(ENGINE)
    report("TensorRT FP16", *collect(lambda b: trt.infer(b)))


if __name__ == "__main__":
    main()
