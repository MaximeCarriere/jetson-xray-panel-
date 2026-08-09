"""XP15 figure — the timing side-channel result.

    ~/xray-venv/bin/python make_figure.py

Reads results/timing_sidechannel.json. Two panels:
  left  — per-category batch-1 latency (mean ± std). If content leaked, the bars would
          separate; they don't (they sit on top of each other).
  right — the positive control: latency by MODEL / input SHAPE, which does separate.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JSON = os.path.join(REPO, "results", "timing_sidechannel.json")
FIG = os.path.join(REPO, "results", "figures", "timing_sidechannel.png")
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "font.size": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})


def main():
    d = json.load(open(JSON))
    cl = d["content_leak"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5),
                                   gridspec_kw={"width_ratios": [2.4, 1]})

    # LEFT — per-category latency, real pathologies then the "unrelated" degenerate inputs
    pc = cl["per_category"]
    real = [c for c in pc if not c.startswith("_")]
    degen = [c for c in pc if c.startswith("_")]
    order = sorted(real) + degen
    labels = [c.replace("_", "") if c.startswith("_") else c for c in order]
    means = [pc[c]["mean_ms"] for c in order]
    stds = [pc[c]["std_ms"] for c in order]
    colors = ["#0072B2"] * len(real) + ["#999999"] * len(degen)
    x = np.arange(len(order))
    axL.bar(x, means, yerr=stds, color=colors, capsize=3, error_kw={"ecolor": MUTED, "lw": 1})
    gm = cl["grand_mean_ms"]
    axL.axhline(gm, color=INK, ls="--", lw=1)
    axL.set_xticks(x); axL.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axL.set_ylabel("batch-1 latency (ms)")
    lo = min(means) - max(stds) * 1.5
    axL.set_ylim(max(0, lo - 1), max(means) + max(stds) * 1.5 + 1)   # zoom in on the spread
    axL.set_title(f"CONTENT does NOT leak — every category (blue) and every unrelated input "
                  f"(grey)\nlands within {cl['spread_pct_of_mean']}% of the mean "
                  f"({cl['spread_across_categories_ms']} ms spread, Kruskal-Wallis "
                  f"p={cl['kruskal_wallis']['p']:.2g})", fontsize=10)
    axL.grid(axis="x", visible=False)

    # RIGHT — the positive control
    ctrl = d["shape_model_control"]
    cx = np.arange(len(ctrl))
    cm = [v["mean_ms"] for v in ctrl.values()]
    cs = [v["std_ms"] for v in ctrl.values()]
    bars = axR.bar(cx, cm, yerr=cs, color=["#D55E00", "#E69F00", "#009E73"], capsize=3,
                   error_kw={"ecolor": MUTED, "lw": 1})
    for b, m in zip(bars, cm):
        axR.text(b.get_x() + b.get_width() / 2, m + 1, f"{m:.0f}", ha="center", fontsize=9)
    axR.set_xticks(cx); axR.set_xticklabels(list(ctrl.keys()), rotation=20, ha="right",
                                            fontsize=8)
    axR.set_ylabel("batch-1 latency (ms)")
    axR.set_title("MODEL / SHAPE does leak\n(what actually is observable)", fontsize=10)
    axR.grid(axis="x", visible=False)

    fig.suptitle("XP15 — is your inference time leaking the input? (Jetson Orin Nano, "
                 "DenseNet-121, batch 1)", fontsize=12, y=1.02)
    fig.text(0.5, -0.03, "Left y-axis is zoomed to the spread: the per-category bars are "
             "indistinguishable. The image CONTENT is invisible to a timing attacker; the "
             "MODEL and INPUT SIZE are not.", ha="center", fontsize=8.5, color=MUTED)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"wrote {os.path.relpath(FIG, REPO)}")


if __name__ == "__main__":
    main()
