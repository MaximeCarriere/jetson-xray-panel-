"""XP15 — timing / energy side-channel: does inference time leak the input?

    ~/xray-venv/bin/python timing_leak.py --per-cat 80 --repeats 8

The security question (Soundes / Pierre's domain — SPA/DPA applied to ML): an attacker who
can only *time* your inference — never see the image — might infer something about it. We
measure whether that channel actually carries information, on the real Jetson.

Three things, one at a time (batch 1 — the realistic single-query serving case):

  A. CONTENT     per-pathology-category latency + energy. Does a "cardiomegaly" image take
                 measurably longer than an "effusion" one?  Plus degenerate inputs (noise,
                 all-black, all-white) as "totally unrelated pictures".
  B. SHAPE/MODEL positive control — DenseNet@224 vs ResNet@512, and different model weights.
                 If the channel is real at all, it shows up here.
  C. ENERGY      sustained per-category mean power -> energy per inference (tegrastats is
                 too coarse to resolve a single 5-50 ms inference, so we integrate a loop).

Measurement hygiene (this is the whole point — it is a side-channel measurement, so the
methodology has to be DPA-grade):
  * lock the clocks (jetson_clocks) and power mode, and record them, so DVFS doesn't add
    input-independent jitter that masquerades as signal;
  * warm up and discard;
  * **interleave categories in a shuffled schedule** so thermal drift over the run cannot
    correlate with a category and fake a leak — the single most important control;
  * report EFFECT SIZE (how many microseconds, as a % of the mean), not just a p-value:
    with tens of thousands of samples any trivial difference is "significant", but only a
    difference an attacker could resolve over a network actually leaks.

Writes results/timing_sidechannel.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib import models as M                              # noqa: E402
from lib.power_logger import PowerLogger                 # noqa: E402
from lib.chest_labels import MEDMNIST_LABELS, xrv_normalize  # noqa: E402


def _preprocess(img_u8, size=224):
    """ChestMNIST 224x224 uint8 -> (1,size,size) FP32 in xrv range. Pure numpy (no cv2):
    the images are already square 224, so no resize/crop is needed."""
    return xrv_normalize(img_u8.astype(np.float32))[None, ...].astype(np.float32)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "timing_sidechannel.json")
DEVICE = "cuda"


@torch.no_grad()
def _infer_once(model, x):
    """One batch-1 forward in FP16 autocast (repo convention), fully synchronized."""
    with torch.autocast("cuda", dtype=torch.float16):
        model(x)
    torch.cuda.synchronize()


def wall_latencies(model, tensors, repeats, schedule):
    """Per-inference WALL latency (ms) — what a timing attacker actually observes.

    `schedule` is a list of indices into `tensors`, pre-shuffled so categories are
    interleaved. Returns {index: [latency_ms, ...]}.
    """
    out = {i: [] for i in range(len(tensors))}
    # warm-up on a few items
    for i in schedule[:20]:
        _infer_once(model, tensors[i])
    for i in schedule:
        x = tensors[i]
        for _ in range(repeats):
            t0 = time.perf_counter()
            _infer_once(model, x)
            out[i].append((time.perf_counter() - t0) * 1e3)
    return out


def summarize(latencies_ms):
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {"mean_ms": round(float(a.mean()), 4), "std_ms": round(float(a.std(ddof=1)), 4),
            "p50_ms": round(float(np.percentile(a, 50)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4), "n": int(a.size)}


# --------------------------------------------------------------------------- data
def category_tensors(per_cat, size=224, seed=0):
    """{category: (k,1,size,size) FP32 GPU tensor} of single-pathology images, plus
    degenerate 'unrelated' inputs (noise / black / white)."""
    from medmnist import ChestMNIST
    ds = ChestMNIST(split="test", size=size, download=True)
    imgs, labels = ds.imgs, ds.labels                    # (N,224,224) u8, (N,14) {0,1}
    rng = np.random.default_rng(seed)

    groups: dict[str, list] = {}
    pos = labels.sum(1)
    # no-finding = all-negative images
    idx_nf = np.flatnonzero(pos == 0)
    if idx_nf.size:
        groups["no_finding"] = list(rng.choice(idx_nf, min(per_cat, idx_nf.size), False))
    # single-pathology images -> that pathology's category (clean labels)
    for c, name in enumerate(MEDMNIST_LABELS):
        idx = np.flatnonzero((labels[:, c] == 1) & (pos == 1))
        if idx.size >= 20:
            groups[name] = list(rng.choice(idx, min(per_cat, idx.size), False))

    tensors: dict[str, torch.Tensor] = {}
    for name, idxs in groups.items():
        arr = np.stack([_preprocess(imgs[i], size) for i in idxs])   # (k,1,size,size)
        tensors[name] = torch.from_numpy(arr).to(DEVICE)

    # "totally unrelated pictures", part 1 — degenerate inputs (nothing like an X-ray).
    k = per_cat
    tensors["_noise"] = ((torch.rand(k, 1, size, size) * 2048) - 1024).to(DEVICE)
    tensors["_black"] = torch.full((k, 1, size, size), -1024.0).to(DEVICE)
    tensors["_white"] = torch.full((k, 1, size, size), 1024.0).to(DEVICE)

    # part 2 — REAL natural images from a totally different domain (CIFAR-10: cars, cats,
    # ships). Resized 32->224 with PIL (cv2 is ABI-broken on this board). If an attacker
    # can't tell a car from a chest X-ray by timing, content really doesn't leak.
    try:
        from PIL import Image
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root=os.path.join(REPO, "results", "raw", "cifar"),
                     train=False, download=True)
        want = {"automobile": "~car", "cat": "~cat", "ship": "~ship"}
        cls = ds.classes
        buckets = {tag: [] for tag in want.values()}
        for img, y in ds:                                # img is a PIL RGB 32x32
            name = cls[y]
            if name in want and len(buckets[want[name]]) < per_cat:
                g = np.array(img.convert("L").resize((size, size), Image.BILINEAR))
                buckets[want[name]].append(xrv_normalize(g.astype(np.float32))[None, ...])
            if all(len(b) >= per_cat for b in buckets.values()):
                break
        for tag, arrs in buckets.items():
            if arrs:
                tensors[tag] = torch.from_numpy(np.stack(arrs).astype(np.float32)).to(DEVICE)
    except Exception as e:                               # torchvision/CIFAR unavailable
        print(f"   (CIFAR unrelated images skipped: {e})", flush=True)
    return tensors


# --------------------------------------------------------------------------- A. content
def content_leak(model, tensors, repeats, seed=0):
    """Per-category batch-1 latency, measured on an interleaved (thermally decontaminated)
    schedule. One flat tensor list + a category tag per row."""
    rows, tags = [], []
    for name, t in tensors.items():
        for j in range(t.shape[0]):
            rows.append(t[j:j + 1]); tags.append(name)
    rng = np.random.default_rng(seed)
    schedule = list(rng.permutation(len(rows)))
    lat = wall_latencies(model, rows, repeats, schedule)

    by_cat: dict[str, list] = {}
    for i, tag in enumerate(tags):
        by_cat.setdefault(tag, []).extend(lat[i])
    per_cat = {c: summarize(v) for c, v in by_cat.items()}

    # effect size + Kruskal-Wallis across the pathology categories only (exclude the
    # degenerate "_" inputs and the CIFAR "~" natural images).
    real = {c: by_cat[c] for c in by_cat if not (c.startswith("_") or c.startswith("~"))}
    means = [per_cat[c]["mean_ms"] for c in real]
    grand = float(np.mean([v for vs in real.values() for v in vs]))
    spread_ms = round(max(means) - min(means), 4)
    from scipy.stats import kruskal
    try:
        kw_h, kw_p = kruskal(*real.values())
    except Exception:
        kw_h, kw_p = float("nan"), float("nan")
    return {
        "per_category": per_cat,
        "grand_mean_ms": round(grand, 4),
        "spread_across_categories_ms": spread_ms,
        "spread_pct_of_mean": round(100 * spread_ms / grand, 3),
        "kruskal_wallis": {"H": round(float(kw_h), 3), "p": float(kw_p),
                           "note": "tests if the per-category latency distributions differ"},
    }


# --------------------------------------------------------------------------- B. control
def shape_model_control(per_cat, repeats, seed=0):
    """Positive control: does latency change with input SHAPE and MODEL? (It should.)"""
    rng = np.random.default_rng(seed)
    configs = [
        ("densenet-nih @224", "densenet121-res224-nih", 224),
        ("densenet-chex @224", "densenet121-res224-chex", 224),
        ("resnet50 @512", M.RESNET, 512),
    ]
    out = {}
    for label, name, size in configs:
        model = M.load_model(name, DEVICE)
        x = ((torch.rand(1, 1, size, size) * 2048) - 1024).to(DEVICE)
        for _ in range(20):
            _infer_once(model, x)
        lat = []
        for _ in range(per_cat * repeats):
            t0 = time.perf_counter()
            _infer_once(model, x)
            lat.append((time.perf_counter() - t0) * 1e3)
        out[label] = {**summarize(lat), "input": f"{size}x{size}", "weights": name}
        del model
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- C. energy
def energy_per_category(model, tensors, seconds=3.0):
    """Sustained per-category power -> energy per inference (tegrastats is too coarse for a
    single inference, so we integrate a loop). Randomised category order, cooldown between."""
    import random
    cats = [c for c in tensors if not c.startswith("_")][:6]     # a representative subset
    random.Random(0).shuffle(cats)
    out = {}
    for c in cats:
        t = tensors[c]
        x = t[:1]
        for _ in range(20):
            _infer_once(model, x)
        with PowerLogger(interval_ms=50) as p:
            t0 = time.perf_counter()
            n = 0
            while time.perf_counter() - t0 < seconds:
                _infer_once(model, x); n += 1
            t1 = time.perf_counter()
        e = p.energy_joules(t0, t1)
        s = p.summary(t0, t1)
        out[c] = {"inferences": n, "seconds": round(t1 - t0, 2),
                  "power_w_mean": s["power_w"]["mean"],
                  "energy_j_total": round(e, 3),
                  "energy_mj_per_inference": round(e / max(1, n) * 1000, 4)}
        time.sleep(2.0)                                  # cooldown between categories
    return out


def read_power_mode():
    import subprocess
    try:
        q = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        mode = [l for l in q.stdout.splitlines() if l.strip()]
        return mode[-1] if mode else "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--per-cat", type=int, default=80, help="images per category")
    a.add_argument("--repeats", type=int, default=8, help="timing repeats per image")
    a.add_argument("--no-energy", action="store_true")
    a.add_argument("--content-only", action="store_true",
                   help="re-measure only Part A; keep B/C from the existing JSON")
    args = a.parse_args()
    if not torch.cuda.is_available():
        print("error: needs CUDA (run on the Jetson)", file=sys.stderr); return 2

    print(f"device={torch.cuda.get_device_name(0)}  power_mode={read_power_mode()}")
    print("loading data + model...", flush=True)
    tensors = category_tensors(args.per_cat)
    model = M.load_model("densenet121-res224-nih", DEVICE)

    print("A. content leak (interleaved, batch 1)...", flush=True)
    content = content_leak(model, tensors, args.repeats)
    print(f"   spread across categories: {content['spread_across_categories_ms']} ms "
          f"({content['spread_pct_of_mean']}% of mean {content['grand_mean_ms']} ms), "
          f"Kruskal-Wallis p={content['kruskal_wallis']['p']:.3g}")

    prev = json.load(open(OUT)) if (args.content_only and os.path.isfile(OUT)) else {}
    if args.content_only:
        print("B/C reused from existing JSON (--content-only)")
        control = prev.get("shape_model_control")
        energy = prev.get("energy_per_category")
    else:
        print("B. shape / model control...", flush=True)
        control = shape_model_control(args.per_cat, args.repeats)
        for k, v in control.items():
            print(f"   {k:22} {v['mean_ms']:.3f} ms")
        energy = None
        if not args.no_energy:
            print("C. energy per category (sustained)...", flush=True)
            energy = energy_per_category(model, tensors)

    out = {
        "experiment": "xp15_timing_sidechannel",
        "device": torch.cuda.get_device_name(0),
        "power_mode": read_power_mode(),
        "model": "densenet121-res224-nih (FP16 autocast, batch 1)",
        "per_category_images": args.per_cat, "timing_repeats": args.repeats,
        "content_leak": content,
        "shape_model_control": control,
        "energy_per_category": energy,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
