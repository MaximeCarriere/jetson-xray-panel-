"""XP14 — Are the pathology probabilities calibrated? (and a double-sigmoid bug it found)

AUROC (all prior experiments) measures only *discrimination* — can the model rank a sick
patient above a healthy one? It says nothing about whether a predicted "0.7" actually means
a 70% chance. That second property is **calibration**, and it's what makes a probability
trustworthy enough to turn into a word ("likely" / "possible", as XP13 does).

We evaluate the DenseNet-nih single-model predictions on 2000 labeled ChestMNIST images
(the exact predictions saved for XP8 in results/tta_preds.npz), pooled over all 14
pathologies, and measure calibration three ways:

  1. **as-stored** — the values XP8 saved. Turns out these were **double-sigmoided**: the
     torchxrayvision model's forward already returns probabilities (sigmoid + op_norm), and
     the eval path applied torch.sigmoid *again*, squashing every value into [0.5, 0.73].
     AUROC never noticed (sigmoid is monotonic -> identical ranking), but it wrecks
     calibration. This is the bug.
  2. **fixed** — recover the model's real probability by inverting that extra sigmoid
     (mathematically identical to removing the double sigmoid from the pipeline).
  3. **temperature-scaled** — fit one scalar T on the logits (p = sigmoid(z / T)) to
     minimise NLL, the standard post-hoc calibration fix.

Reports Expected Calibration Error (ECE) and Brier score for each, plus the reliability
curve, to results/calibration.json.  Pure numpy/scipy — runs on any machine, no board.

    python calibration.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EPS = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 12):
    """Equal-width reliability curve + ECE (expected calibration error)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    curve, ece, n = [], 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = float(p[m].mean()), float(y[m].mean()), m.sum() / n
        ece += w * abs(acc - conf)
        curve.append({"conf": round(conf, 4), "acc": round(acc, 4), "n": int(m.sum())})
    return curve, float(ece)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> None:
    d = np.load(os.path.join(REPO, "results", "tta_preds.npz"))
    y = d["labels"].ravel().astype(np.float64)      # (28000,) {0,1}
    p_raw = d["single"].ravel()

    # The single-model column is exactly recoverable: stored = sigmoid(model_prob), so
    # model_prob = logit(stored). Detect whether we're looking at the double-sigmoided
    # data (everything squashed into [0.5, 0.73]) or an already-fixed rerun, and set up
    # both the buggy and the corrected probability vectors either way.
    if p_raw.max() < 0.80:                            # double-sigmoided
        p_bug, p_fixed = p_raw, _logit(p_raw)
    else:                                            # already fixed -> reconstruct the bug
        p_fixed, p_bug = p_raw, _sigmoid(p_raw)
    z = _logit(p_fixed)                              # logits, for temperature scaling

    # (3) temperature scaling: fit T>0 minimising NLL of sigmoid(z / T).
    def nll(T: float) -> float:
        p = np.clip(_sigmoid(z / T), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T = float(res.x)
    p_temp = _sigmoid(z / T)

    variants = {"stored_double_sigmoid": p_bug, "fixed": p_fixed, "temperature": p_temp}
    out = {
        "n_pairs": int(y.size),
        "prevalence": round(float(y.mean()), 4),
        "temperature": round(T, 3),
        "ece": {}, "brier": {}, "mean_pred": {}, "reliability": {},
    }
    for name, p in variants.items():
        curve, ece = reliability(y, p)
        out["ece"][name] = round(ece, 4)
        out["brier"][name] = round(brier(y, p), 4)
        out["mean_pred"][name] = round(float(p.mean()), 4)
        out["reliability"][name] = curve

    path = os.path.join(REPO, "results", "calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"prevalence (true positive rate): {out['prevalence']:.4f}")
    print(f"fitted temperature T = {T:.3f}")
    print(f"{'variant':<24} {'mean pred':>10} {'ECE':>8} {'Brier':>8}")
    for name in variants:
        print(f"{name:<24} {out['mean_pred'][name]:>10.4f} "
              f"{out['ece'][name]:>8.4f} {out['brier'][name]:>8.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
