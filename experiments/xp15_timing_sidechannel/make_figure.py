"""XP15 figure — the timing side-channel result, one clean picture.

    python make_figure.py           # reads results/timing_sidechannel.json (no board needed)

Left  : per-inference latency for every input — 15 chest-X-ray pathologies, 3 degenerate
        inputs, 3 CIFAR photos — as a horizontal dot plot. They all land on one line: the
        image CONTENT does not leak.
Right : the same measurement across models / input sizes — which pipeline ran DOES leak.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JSON = os.path.join(REPO, "results", "timing_sidechannel.json")
FIG = os.path.join(REPO, "results", "figures", "timing_sidechannel.png")

XRAY, DEGEN, CIFAR, MODEL = "#0072B2", "#999999", "#D55E00", "#333333"
INK, MUTED, GRID = "#222222", "#666666", "#e2e2e2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.titlecolor": INK, "font.size": 11, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})


def main():
    d = json.load(open(JSON))
    cl = d["content_leak"]
    pc = cl["per_category"]
    gm = cl["grand_mean_ms"]

    path = sorted(c for c in pc if not c.startswith(("_", "~")))
    degen = [c for c in pc if c.startswith("_")]
    cifar = [c for c in pc if c.startswith("~")]
    order = path + degen + cifar                       # bottom-to-top after we flip
    order = order[::-1]

    def lab(c):
        return (c[1:] + "  (noise/black/white)" if c.startswith("_") else
                c[1:] + "  (CIFAR photo)" if c.startswith("~") else c)

    def col(c):
        return DEGEN if c.startswith("_") else CIFAR if c.startswith("~") else XRAY

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.32)
    axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])

    # -------- LEFT: content — a tight vertical line = no leak
    y = np.arange(len(order))
    means = np.array([pc[c]["mean_ms"] for c in order])
    stds = np.array([pc[c]["std_ms"] for c in order])
    axL.axvspan(means.min(), means.max(), color="#0072B2", alpha=0.07, zorder=0)
    axL.axvline(gm, color=INK, ls="--", lw=1.2, zorder=1)
    for yi, c, m, s in zip(y, order, means, stds):
        axL.errorbar(m, yi, xerr=s, fmt="o", ms=7, color=col(c), ecolor=MUTED,
                     elinewidth=1.4, capsize=3, zorder=3)
    axL.set_yticks(y); axL.set_yticklabels([lab(c) for c in order], fontsize=9)
    axL.set_xlim(gm - 1.0, gm + 1.0)
    axL.set_xlabel("per-inference latency (ms)  ·  batch 1")
    axL.set_title("What's IN the image doesn't leak\n"
                  f"every input lands within {cl['spread_across_categories_ms']:.3f} ms "
                  f"({cl['spread_pct_of_mean']}% of the mean)", fontsize=12)
    axL.legend(handles=[Patch(color=XRAY, label="chest X-ray (15 pathologies)"),
                        Patch(color=DEGEN, label="degenerate (noise / black / white)"),
                        Patch(color=CIFAR, label="CIFAR photo (car / cat / ship)")],
               loc="lower right", fontsize=9, framealpha=0.95)
    axL.grid(axis="y", visible=False)
    axL.annotate(f"grand mean {gm:.2f} ms", xy=(gm, len(order) - 0.5),
                 xytext=(gm + 0.12, len(order) - 0.6), fontsize=8.5, color=MUTED)

    # -------- RIGHT: model / resolution — clearly separated = the real leak
    ctrl = d["shape_model_control"]
    names = list(ctrl); cm = [ctrl[n]["mean_ms"] for n in names]
    cs = [ctrl[n]["std_ms"] for n in names]
    yy = np.arange(len(names))[::-1]
    bars = axR.barh(yy, cm, xerr=cs, color=[MODEL, "#8a8a8a", "#D55E00"], height=0.6,
                    error_kw={"ecolor": MUTED, "lw": 1.2}, zorder=3)
    for b, m in zip(bars, cm):
        axR.text(m + 1.2, b.get_y() + b.get_height() / 2, f"{m:.0f} ms", va="center",
                 fontsize=10, fontweight="bold")
    axR.set_yticks(yy); axR.set_yticklabels(names, fontsize=9)
    axR.set_xlim(0, 60); axR.set_xlabel("per-inference latency (ms)")
    axR.set_title("Which MODEL / resolution ran DOES leak\n(a clean 2× gap)", fontsize=12)
    axR.grid(axis="y", visible=False)

    fig.suptitle("XP15 — is your inference time leaking the input?  "
                 "(Jetson Orin Nano, DenseNet-121)", fontsize=13, y=1.01, fontweight="bold")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print(f"wrote {os.path.relpath(FIG, REPO)}")


if __name__ == "__main__":
    main()
