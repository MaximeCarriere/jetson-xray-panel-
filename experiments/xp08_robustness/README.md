# XP8 — TTA / ensemble robustness

Spend the spare GPU capacity on *reliability* instead of speed. Evaluated on
**ChestMNIST-224** (NIH, 2000 labeled test images, auto-downloaded — no Kaggle),
macro-AUROC over 14 pathologies.

## What are TTA and Ensemble?

Two different ways to average away individual-prediction errors — they vary *different*
things:

- **TTA (Test-Time Augmentation):** **one model**, and **the same image shown in several
  mildly-augmented views** (small rotations, contrast/brightness shifts). The model
  classifies each view, and the predictions are averaged. Idea: averaging over
  slightly-different *valid* views cancels single-pass noise. *(It varies the **image**.)*
- **Ensemble:** the **same, unmodified image** fed to **several different models** (here,
  DenseNet-121s trained on different datasets — NIH, all, MIMIC). Their predictions are
  averaged. Idea: models trained differently make different mistakes, which cancel.
  *(It varies the **model**.)*

The picture below is generated from a real ChestMNIST image (`illustrate.py`) — the top
row is literally the same X-ray in 5 augmented views (one model); the bottom is one
image through three models:

![TTA vs Ensemble](../../results/figures/tta_vs_ensemble.png)

## Result
| Method | AUROC (± bootstrap SE) | Δ |
|---|---:|---:|
| Single pass | 0.7405 ± 0.0134 | — |
| TTA (5 augmented views) | 0.7375 ± 0.0135 | −0.003 |
| **Ensemble (3 different-dataset DenseNets)** | **0.7621 ± 0.0137** | **+0.022** |

- **Ensembling helps** (+0.022) — but with proper error bars (±1 bootstrap SE) the
  per-estimate CIs overlap, so at 2000 images the gain is **suggestive, not
  conclusive**; a paired test or more data would firm it up. Honest, not overclaimed.
- **Naive TTA does not help** (−0.003, squarely within noise) — matches the literature.
- **Cost ~free:** 5 views / 3 models run as one batch cost ~the same as a single pass —
  the spare capacity absorbs it.

![tta robustness](../../results/figures/tta_robustness.png)

## Run
```bash
setsid bash run_tta.sh --n 2000 --views 5
```

## Files
`tta_experiment.py` · `run_tta.sh`. Label map + AUROC in `lib/chest_labels.py`.
Data `../../results/tta_bench.json`.
