"""Shared ChestMNIST label mapping + macro-AUROC (used by xp06 and xp08).

ChestMNIST uses the 14 NIH ChestX-ray14 labels; torchxrayvision models emit their
pathologies in a different order (and name pleural thickening differently), so we
map model output columns onto the ChestMNIST label columns by name.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

# ChestMNIST label order (from medmnist INFO).
MEDMNIST_LABELS = ["atelectasis", "cardiomegaly", "effusion", "infiltration",
                   "mass", "nodule", "pneumonia", "pneumothorax", "consolidation",
                   "edema", "emphysema", "fibrosis", "pleural", "hernia"]


def norm_name(p: str) -> str:
    n = p.lower()
    return "pleural" if n.startswith("pleural") else n


def xrv_normalize(img255):
    """[0,255] -> xrv range [-1024,1024] (matches xrv.datasets.normalize)."""
    return (2.0 * (img255 / 255.0) - 1.0) * 1024.0


def col_map(model):
    """Map a model's output columns -> ChestMNIST label columns by name."""
    pm = []
    for j, path in enumerate(model.pathologies):
        if not path:
            continue
        name = norm_name(path)
        if name in MEDMNIST_LABELS:
            pm.append((j, MEDMNIST_LABELS.index(name)))
    return pm


def macro_auroc(y_true, y_score):
    """Mean AUROC over labels that have both classes present. Returns (auc, n_labels)."""
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.min() == yt.max() or np.isnan(y_score[:, c]).any():
            continue
        aucs.append(roc_auc_score(yt, y_score[:, c]))
    return float(np.mean(aucs)), len(aucs)
