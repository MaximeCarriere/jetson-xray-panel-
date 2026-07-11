# XP8 — TTA / ensemble robustness

Spend the spare GPU capacity on *reliability* instead of speed. Evaluated on
**ChestMNIST-224** (NIH, 2000 labeled test images, auto-downloaded — no Kaggle),
macro-AUROC over 14 pathologies.

## Result
| Method | AUROC | Δ |
|---|---:|---:|
| Single pass | 0.7405 | — |
| TTA (5 augmented views) | 0.7375 | −0.003 |
| **Ensemble (3 different-dataset DenseNets)** | **0.7621** | **+0.022** |

- **Ensembling helps** (+0.022); **naive TTA does not** (−0.003, honest negative —
  matches the literature that augmentation-averaging isn't automatically beneficial).
- **Cost ~free:** 5 views / 3 models run as one batch cost 49.2 ms vs 49.0 ms single —
  the spare capacity absorbs it.

![tta robustness](../../results/figures/tta_robustness.png)

## Run
```bash
setsid bash run_tta.sh --n 2000 --views 5
```

## Files
`tta_experiment.py` · `run_tta.sh`. Label map + AUROC in `lib/chest_labels.py`.
Data `../../results/tta_bench.json`.
