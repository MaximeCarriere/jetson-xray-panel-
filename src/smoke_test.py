"""Day-1/2 deliverable: prove the environment end-to-end.

Confirms CUDA is live, loads one pretrained model, runs a single FP16 (autocast)
inference, and prints the pathology probabilities so we can eyeball that they are
in range. Run on the board:

    ~/xray-venv/bin/python ~/jetson-xray-panel/src/smoke_test.py
"""
from __future__ import annotations

import numpy as np
import torch

import models
import utils


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available!"
    dev = torch.cuda.get_device_name(0)
    print(f"CUDA OK on '{dev}', torch {torch.__version__}")

    utils.seed_everything(0)
    name = "densenet121-res224-all"
    size = models.input_size_for(name)
    model = models.load_model(name, device="cuda")
    print(f"Loaded {name}: {len(model.pathologies)} pathologies, input {size}x{size}")

    # One synthetic image (content irrelevant for the smoke test / for timing).
    x = utils.build_input_pool(size=size, n=1, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(x)[0].float().cpu().numpy()

    assert out.shape == (len(model.pathologies),)
    assert np.isfinite(out).all() and 0.0 <= out.min() and out.max() <= 1.0, \
        "probabilities out of [0,1] range"

    top = np.argsort(out)[::-1][:5]
    print(f"Inference OK — prob range [{out.min():.3f}, {out.max():.3f}]")
    print("Top-5 pathologies:")
    for i in top:
        print(f"    {model.pathologies[i]:<28} {out[i]:.3f}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
