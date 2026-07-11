"""Small statistics helpers for honest error bars.

Two kinds of uncertainty appear in this project:
  * run-to-run variance in a throughput measurement -> repeat the run N times and
    report mean ± standard error (SE = sample_std / sqrt(N));
  * uncertainty of an AUROC computed on a *fixed* test set with a *deterministic*
    model -> re-running gives the identical number, so the error bar must come from
    resampling the test set (bootstrap), not from repeating the run.
"""
from __future__ import annotations

import numpy as np


def agg(vals) -> dict:
    """mean / sample-std / standard-error / n over a list of measurements."""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "se": 0.0, "n": 0}
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(a.mean()), "std": std, "se": std / np.sqrt(n), "n": n}


def bootstrap_auroc(y_true, y_score, auroc_fn, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the macro-AUROC over the test set.

    ``auroc_fn(y_true, y_score) -> (auc, n_labels)``. Resamples rows with
    replacement ``n_boot`` times; returns the point estimate plus the bootstrap
    mean / standard error / 95% percentile interval.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point, _ = auroc_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            auc, _ = auroc_fn(y_true[idx], y_score[idx])
            if np.isfinite(auc):
                boots.append(auc)
        except Exception:
            continue
    b = np.asarray(boots)
    return {
        "auroc": float(point),
        "se": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None],
        "n_boot": len(b),
    }
