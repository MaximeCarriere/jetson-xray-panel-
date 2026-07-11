"""Evaluate a TensorRT engine's macro-AUROC on ChestMNIST-224 test images.

Runs the engine over N labeled NIH chest X-rays and reports macro-AUROC over the
14 pathologies, so FP16 and INT8 engines can be compared to the PyTorch baseline.

    ~/xray-venv/bin/python trt_eval_auroc.py ~/densenet_nih_fp16.engine
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

import numpy as np
import torch

import models
import chest_labels as cl              # shared label map + macro-AUROC
from trt_runner import TRTModel


def eval_engine(engine_path: str, n: int = 2000) -> float:
    from medmnist import ChestMNIST
    ds = ChestMNIST(split="test", size=224, download=True)
    imgs = ds.imgs[:n].astype(np.float32)
    labels = ds.labels[:n].astype(np.int64)
    x = torch.from_numpy((2.0 * (imgs / 255.0) - 1.0) * 1024.0)[:, None].cuda()

    # Engine outputs 14 logits in the nih model's pathology order; map to ChestMNIST.
    nih = models.load_model("densenet121-res224-nih", "cuda")
    cmap = cl.col_map(nih)

    trt_model = TRTModel(engine_path)
    preds = np.full((n, len(cl.MEDMNIST_LABELS)), np.nan)
    bs = 16          # must stay within the engine's max batch profile (16)
    t0 = time.perf_counter()
    for i in range(0, n, bs):
        out = trt_model.infer(x[i:i + bs].contiguous())
        probs = torch.sigmoid(out).cpu().numpy()
        for mc, dc in cmap:
            preds[i:i + bs, dc] = probs[:, mc]
    dt = time.perf_counter() - t0

    auc, k = cl.macro_auroc(labels, preds)
    from stats import bootstrap_auroc
    bs = bootstrap_auroc(labels, preds, cl.macro_auroc)
    print(f"{engine_path.split('/')[-1]:28s} macro-AUROC {auc:.4f} ± {bs['se']:.4f} "
          f"(95% CI [{bs['ci95'][0]:.4f}, {bs['ci95'][1]:.4f}], {k} labels, {n} imgs, {n/dt:.0f} img/s)")
    return {"auroc": auc, "se": bs["se"], "ci95": bs["ci95"]}


if __name__ == "__main__":
    for p in sys.argv[1:] or ["/home/a/densenet_nih_fp16.engine"]:
        eval_engine(p)
