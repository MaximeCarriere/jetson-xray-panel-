"""Test-time augmentation (TTA) + ensemble robustness (Section 4 stretch, H6).

Question: can we spend the box's spare concurrent capacity to make each diagnosis
more RELIABLE (higher AUROC), not just faster? We compare, on real labeled chest
X-rays (ChestMNIST-224 = NIH ChestX-ray14, auto-downloaded from Zenodo, no auth):

  * single-pass  — one forward pass per image (baseline)
  * TTA          — K mild label-preserving views (rotation, contrast, brightness),
                   predictions averaged
  * ensemble     — several different-dataset DenseNets, predictions averaged

and report macro-AUROC over the 14 pathologies for each, plus the systems cost.

    ~/xray-venv/bin/python tta_experiment.py --n 2000 --views 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

import numpy as np
import torch
import torchvision.transforms.functional as TF

import models
from chest_labels import MEDMNIST_LABELS, col_map, macro_auroc, xrv_normalize


def _augment(batch255: torch.Tensor, angle: float, contrast: float,
             bright: float, scale: float = 1.0) -> torch.Tensor:
    """One label-preserving view of a [0,255] batch (N,1,H,W).

    Transforms are chosen to be *valid* variations of a real chest X-ray:
      * brightness / contrast  — X-ray exposure varies shot to shot (most realistic);
      * mild rotation ±(4-8°)  — patients aren't perfectly upright (positioning);
      * mild scale 0.94-1.07   — source-to-detector distance / zoom.
    NO horizontal flip — it would flip cardiac/situs laterality (not label-preserving).
    """
    x = batch255
    if bright != 1.0:
        x = (x * bright).clamp(0, 255)
    if contrast != 1.0:
        m = x.mean(dim=(-1, -2), keepdim=True)
        x = ((x - m) * contrast + m).clamp(0, 255)
    if angle != 0.0 or scale != 1.0:
        x = TF.affine(x, angle=angle, translate=[0, 0], scale=scale, shear=[0.0], fill=0.0)
    return x


# Fixed TTA view bank as (angle, contrast, bright, scale). View 0 = untouched image
# (the single-pass baseline). First 5 are used for --views 5; all 10 for --views 10.
VIEW_BANK = [
    (0.0, 1.00, 1.00, 1.00),     # original
    (6.0, 1.00, 1.00, 1.00),     # rotate +6°
    (-6.0, 1.00, 1.00, 1.00),    # rotate -6°
    (0.0, 1.15, 1.00, 1.00),     # contrast +15%
    (0.0, 0.88, 1.05, 1.00),     # contrast -12% / bright +5%
    (0.0, 1.00, 1.00, 1.06),     # zoom in 6%
    (0.0, 1.00, 1.00, 0.94),     # zoom out 6%
    (4.0, 1.08, 0.97, 1.00),     # rot +4° / contrast +8% / bright -3%
    (-4.0, 0.95, 1.04, 1.00),    # rot -4° / contrast -5% / bright +4%
    (0.0, 1.10, 0.96, 1.03),     # contrast +10% / bright -4% / zoom +3%
]


def _predict(model, batch255: torch.Tensor, col_map, n_labels, bs=64) -> np.ndarray:
    """Run model over a [0,255] batch, return (N,14) probs mapped to MEDMNIST order."""
    out = np.full((batch255.shape[0], n_labels), np.nan, dtype=np.float64)
    for i in range(0, batch255.shape[0], bs):
        chunk = xrv_normalize(batch255[i:i + bs])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(chunk)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for model_col, med_col in col_map:
            out[i:i + bs, med_col] = probs[:, model_col]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="test images to evaluate")
    ap.add_argument("--views", type=int, default=5, help="TTA views (incl. original)")
    ap.add_argument("--export-preds", default=None, help="save raw predictions (.npz) for ROC")
    args = ap.parse_args()

    from medmnist import ChestMNIST
    print("Loading ChestMNIST-224 test split (downloads once)…", flush=True)
    ds = ChestMNIST(split="test", size=224, download=True)
    imgs, labels = ds.imgs, ds.labels               # (N,224,224) uint8, (N,14) {0,1}
    idx = np.arange(min(args.n, len(imgs)))
    imgs, labels = imgs[idx], labels[idx].astype(np.int64)
    print(f"  {len(imgs)} images, {labels.shape[1]} labels", flush=True)

    # (N,1,224,224) float [0,255] on GPU.
    batch = torch.from_numpy(imgs).float().unsqueeze(1).cuda()
    n_lab = len(MEDMNIST_LABELS)

    nih = models.load_model("densenet121-res224-nih", "cuda")
    cmap = col_map(nih)

    # 1) single-pass (view 0 only)
    single = _predict(nih, batch, cmap, n_lab)
    auc_single, k = macro_auroc(labels, single)

    # 2) TTA — average K views
    views = VIEW_BANK[:args.views]
    acc = np.zeros_like(single)
    for view in views:
        acc += _predict(nih, _augment(batch, *view), cmap, n_lab)
    tta = acc / len(views)
    auc_tta, _ = macro_auroc(labels, tta)

    # 3) ensemble — average different-dataset DenseNets (single-pass each)
    ens_models = ["densenet121-res224-all", "densenet121-res224-nih",
                  "densenet121-res224-mimic_ch"]
    ens_acc = np.zeros_like(single)
    ens_cnt = np.zeros_like(single)
    for name in ens_models:
        m = models.load_model(name, "cuda")
        p = _predict(m, batch, col_map(m), n_lab)
        mask = ~np.isnan(p)
        ens_acc[mask] += p[mask]
        ens_cnt[mask] += 1
    ensemble = np.where(ens_cnt > 0, ens_acc / np.maximum(ens_cnt, 1), np.nan)
    auc_ens, _ = macro_auroc(labels, ensemble)

    if args.export_preds:
        np.savez(args.export_preds, labels=labels, single=single, tta=tta, ensemble=ensemble)
        print(f"exported predictions -> {args.export_preds}")

    # Bootstrap the AUROC over the test set for an honest error bar (the model is
    # deterministic, so re-running is identical — uncertainty is sampling, not runs).
    from stats import bootstrap_auroc
    import json
    bs_single = bootstrap_auroc(labels, single, macro_auroc)
    bs_tta = bootstrap_auroc(labels, tta, macro_auroc)
    bs_ens = bootstrap_auroc(labels, ensemble, macro_auroc)

    print(f"\n=== Robustness (macro-AUROC over {k} pathologies, {len(imgs)} images) ===")
    print(f"  single-pass          : {auc_single:.4f} ± {bs_single['se']:.4f}")
    print(f"  TTA ({len(views)} views)         : {auc_tta:.4f} ± {bs_tta['se']:.4f}  ({auc_tta-auc_single:+.4f})")
    print(f"  ensemble ({len(ens_models)} models)   : {auc_ens:.4f} ± {bs_ens['se']:.4f}  ({auc_ens-auc_single:+.4f})")

    # 4) systems cost — TTA views as ONE batch cost ~the same as one image
    #    (batching fills spare capacity), so robustness is nearly free.
    def _time(fn, iters=30, warm=5):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    one = batch[:1]
    kviews = torch.cat([_augment(one, *v) for v in views], 0)  # (K,1,224,224)

    def single_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            nih(xrv_normalize(one))

    def tta_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            nih(xrv_normalize(kviews))

    t_single = _time(single_fn)
    t_tta = _time(tta_fn)
    print(f"\n=== Systems cost (per image) ===")
    print(f"  single pass         : {t_single:.1f} ms")
    print(f"  TTA {len(views)} views (1 batch): {t_tta:.1f} ms  "
          f"({t_tta/t_single:.2f}x latency for {len(views)}x the passes)")

    with open("/home/a/jetson-xray-panel/results/tta_bench.json", "w") as f:
        json.dump({"n_images": len(imgs), "views": len(views),
                   "ensemble_models": len(ens_models),
                   "auroc": {"single": round(auc_single, 4), "tta_5views": round(auc_tta, 4),
                             "ensemble_3models": round(auc_ens, 4)},
                   "auroc_se": {"single": round(bs_single["se"], 4),
                                "tta_5views": round(bs_tta["se"], 4),
                                "ensemble_3models": round(bs_ens["se"], 4)},
                   "latency_ms": {"single": round(t_single, 1),
                                  "tta_5views_one_batch": round(t_tta, 1)}}, f, indent=2)
    print("wrote results/tta_bench.json")


if __name__ == "__main__":
    main()
