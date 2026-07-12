"""Export images that show what TTA vs Ensemble actually do, for the README figure.

TTA  = one image -> the 5 augmented views (same model sees all of them).
Ensemble = one unmodified image -> the 3 different models.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

import numpy as np
import torch
from PIL import Image

from tta_experiment import VIEW_BANK, _augment

VIEW_NAMES = ["original", "rotate +7°", "rotate −7°", "contrast +15%", "bright +5% / contrast −10%"]


def datauri(arr: np.ndarray) -> str:
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(1e-3, hi - lo) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    from medmnist import ChestMNIST
    ds = ChestMNIST(split="test", size=224, download=True)
    img = ds.imgs[296].astype(np.float32)           # a clear effusion case
    batch = torch.from_numpy(img)[None, None]        # (1,1,224,224) float [0,255]

    views = []
    for (ang, con, bri), name in zip(VIEW_BANK[:5], VIEW_NAMES):
        v = _augment(batch, ang, con, bri)[0, 0].numpy()
        views.append({"name": name, "img": datauri(v)})

    out = {
        "original": datauri(img),
        "tta_views": views,
        "ensemble_models": ["densenet121-res224-nih (NIH)",
                            "densenet121-res224-all (all datasets)",
                            "densenet121-res224-mimic_ch (MIMIC)"],
    }
    path = "/home/a/jetson-xray-panel/results/tta_views.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(views)} TTA views + original -> {path}")


if __name__ == "__main__":
    main()
