"""Pretrained chest X-ray model registry (torchxrayvision).

Every model is a DenseNet-121 (or a ResNet-50 stressor) pretrained on public
chest X-ray datasets. We use them AS-IS — no retraining. The point of this
project is the *systems* behaviour (throughput / power / concurrency), not
accuracy, so we never touch the weights.

Precision note: run inference in FP16 via ``torch.autocast`` at call time, NOT
``model.half()``. Calling ``.half()`` on the whole module breaks torchxrayvision's
``op_norm`` post-processing (its ``op_threshs`` buffer stays FP32 -> dtype error).
Autocast keeps the params/buffers FP32 and only casts the heavy conv/matmul work.
"""
from __future__ import annotations

import torch
import torchxrayvision as xrv

# The seven DenseNet-121 variants below are trained on DIFFERENT datasets, so
# they are genuinely different sets of weights. This matters for the concurrency
# demo: different weights CANNOT be fused into one batched forward pass, so a
# multi-disease "panel" of different models is exactly the case where concurrent
# execution is the only option (you can't batch your way out of it).
DENSENET_VARIANTS = [
    "densenet121-res224-all",       # trained on the union of all datasets
    "densenet121-res224-nih",       # NIH ChestX-ray14
    "densenet121-res224-pc",        # PadChest
    "densenet121-res224-chex",      # CheXpert
    "densenet121-res224-rsna",      # RSNA Pneumonia
    "densenet121-res224-mimic_ch",  # MIMIC-CXR (CheXpert labels)
    "densenet121-res224-mimic_nb",  # MIMIC-CXR (NegBio labels)
]

# Heavier, higher-resolution model — used to stress the GPU harder in the ramp.
RESNET = "resnet50-res512-all"


def input_size_for(name: str) -> int:
    """Square input resolution (pixels) the given model expects."""
    return 512 if name.startswith("resnet") else 224


def load_model(name: str, device: str = "cuda") -> torch.nn.Module:
    """Load a pretrained model by weight name, in eval mode, on ``device`` (FP32).

    Weights are downloaded and cached under ~/.torchxrayvision on first use.
    """
    if name.startswith("resnet"):
        model = xrv.models.ResNet(weights=name)
    else:
        model = xrv.models.DenseNet(weights=name)
    return model.eval().to(device)


def distinct_panel(n: int) -> list[str]:
    """Return ``n`` distinct model names for the different-models concurrent case.

    Cycles through the DenseNet variants (and appends the ResNet stressor) so we
    can request more concurrent models than we have unique DenseNets, while
    keeping them as distinct as possible.
    """
    pool = DENSENET_VARIANTS + [RESNET]
    return [pool[i % len(pool)] for i in range(n)]
