"""Generate the committed figures from results/raw/*.json (Section 8 of PLAN.md).

Design follows the dataviz method: metrics of different scale become small
multiples (never a dual-axis chart), categorical hues are assigned in fixed
order, marks are thin, grid/axes recede, and single-series panels are titled
rather than legended.

Run from the repo root (on the Mac, after pulling results back):

    python analysis/make_figures.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "results", "raw")
FIG = os.path.join(REPO, "results", "figures")

# Validated dataviz categorical palette (light mode), assigned in fixed order.
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e2"


def _load() -> list[dict]:
    recs = []
    for p in glob.glob(os.path.join(RAW, "*.json")):
        with open(p) as f:
            recs.append(json.load(f))
    return recs


def _by_config(recs: list[dict]) -> dict[str, dict]:
    """Group records by config; return mean/std of key metrics across repeats."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[r["config"]].append(r)

    def agg(rs, path):
        vals = []
        for r in rs:
            v = r
            for k in path:
                v = v[k]
            if v is not None:
                vals.append(v)
        a = np.asarray(vals, dtype=float)
        return (float(a.mean()), float(a.std())) if len(a) else (np.nan, 0.0)

    out = {}
    for cfg, rs in groups.items():
        out[cfg] = {
            "regime": rs[0]["regime"],
            "case": rs[0].get("concurrency_case"),   # same | diff | ramp | None
            "batch_size": rs[0]["batch_size"],
            "n_models": rs[0]["n_models"],
            "throughput": agg(rs, ["throughput_ips"]),
            "power": agg(rs, ["power_w", "mean"]),
            "efficiency": agg(rs, ["throughput_per_watt"]),
            "gpu_util": agg(rs, ["gpu_util_pct"]),
            "n_repeats": len(rs),
        }
    return out


# Series identity -> (label, color). "Degree of parallelism" x-axis unifies them:
# batch size for batched, #concurrent models for concurrent, 1 for sequential.
SERIES = {
    "batched":         ("Batched (1 model)", BLUE),
    "concurrent_same": ("Concurrent, same model", AQUA),
    "concurrent_diff": ("Concurrent, different models (panel)", YELLOW),
}


def _series(cfg: dict) -> dict[str, list[tuple]]:
    """Group configs into plot series keyed by identity, each a list of
    (x, mean, std) sorted by x, for throughput / power / efficiency."""
    out: dict[str, list] = {k: [] for k in SERIES}
    seq_x1 = None
    for name, c in cfg.items():
        if c["regime"] == "sequential":
            seq_x1 = c
            continue
        if c["regime"] == "batched":
            key, x = "batched", c["batch_size"]
        elif c["regime"] == "concurrent":
            if c["batch_size"] != 1:      # concurrent-batched handled separately
                continue
            key = "concurrent_diff" if c["case"] == "diff" else "concurrent_same"
            x = c["n_models"]
        else:
            continue
        out[key].append((x, c))
    for key in out:
        out[key].sort(key=lambda t: t[0])
    return out, seq_x1


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def fig_batching(cfg: dict) -> None:
    """Single-model batching story: throughput / power / efficiency vs batch size."""
    # Batch series = S1 (batch 1) + B2/B4/B8, ordered by batch size.
    keys = [k for k in cfg if cfg[k]["regime"] in ("sequential", "batched")]
    keys = sorted(keys, key=lambda k: cfg[k]["batch_size"])
    if not keys:
        return
    # Plot on the TRUE batch-size axis (not equal-spaced), so a relationship that
    # is linear in batch size reads as a straight line instead of a fake curve.
    bs = [cfg[k]["batch_size"] for k in keys]
    x = np.asarray(bs, dtype=float)

    panels = [
        ("Throughput", "images / second", "throughput", BLUE),
        ("Power draw", "watts (total, VDD_IN)", "power", YELLOW),
        ("Efficiency", "images / second / watt", "efficiency", AQUA),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=SURFACE)
    for ax, (title, ylab, metric, color) in zip(axes, panels):
        means = [cfg[k][metric][0] for k in keys]
        stds = [cfg[k][metric][1] for k in keys]
        _style(ax)
        ax.errorbar(x, means, yerr=stds, color=color, linewidth=2,
                    marker="o", markersize=7, capsize=3, zorder=3)
        for xi, m in zip(x, means):                      # direct value labels
            ax.annotate(f"{m:.1f}", (xi, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(ylab, color=INK2, fontsize=10)
        ax.set_xlabel("batch size", color=INK2, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(b)) for b in bs])
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, max(bs) * 1.12)

    fig.suptitle("Single-model batching baseline — DenseNet-121, Jetson Orin Nano Super (MAXN_SUPER)",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "batching_baseline.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def _plot_series(ax, series, seq_x1, metric, ylabel, show_seq=True):
    """Plot each regime's series of (x, cfg) as a line for the given metric."""
    for key, (label, color) in SERIES.items():
        pts = series[key]
        if not pts:
            continue
        xs = [x for x, _ in pts]
        means = [c[metric][0] for _, c in pts]
        stds = [c[metric][1] for _, c in pts]
        ax.errorbar(xs, means, yerr=stds, color=color, linewidth=2, marker="o",
                    markersize=6, capsize=3, zorder=3, label=label)
    if show_seq and seq_x1 is not None:
        y = seq_x1[metric][0]
        ax.axhline(y, color=INK2, linewidth=1.2, linestyle="--", zorder=2)
        ax.annotate(f"sequential (1 model): {y:.1f}", (0.02, y),
                    xycoords=("axes fraction", "data"), color=INK2, fontsize=8.5,
                    va="bottom")
    _style(ax)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_xlabel("degree of parallelism  (batch size  /  # concurrent models)",
                  color=INK2, fontsize=9.5)
    ax.set_ylim(bottom=0)


def fig_scaling(cfg: dict) -> None:
    """Headline: throughput vs degree of parallelism, all regimes overlaid."""
    series, seq_x1 = _series(cfg)
    if not any(series.values()):
        return
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    _plot_series(ax, series, seq_x1, "throughput", "images / second")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left",
              labelcolor=INK, bbox_to_anchor=(0.0, 0.98))
    ax.set_title("Throughput scaling: batching vs concurrency\nJetson Orin Nano Super (MAXN_SUPER, MPS)",
                 color=INK, fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    out = os.path.join(FIG, "throughput_scaling.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def fig_power_efficiency(cfg: dict) -> None:
    """Power draw and efficiency (img/s/W) vs degree of parallelism."""
    series, seq_x1 = _series(cfg)
    if not any(series.values()):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    _plot_series(a1, series, seq_x1, "power", "watts (total, VDD_IN)")
    a1.set_title("Power draw", color=INK, fontsize=12, fontweight="bold", loc="left")
    _plot_series(a2, series, seq_x1, "efficiency", "images / second / watt")
    a2.set_title("Energy efficiency  (higher = better)", color=INK, fontsize=12,
                 fontweight="bold", loc="left")
    a2.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle("Power & efficiency vs parallelism — Jetson Orin Nano Super (MAXN_SUPER, MPS)",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG, "power_efficiency.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


VIOLET, RED = "#4a3aa7", "#e34948"


def fig_saturation(cfg: dict) -> None:
    """Throughput vs images-in-flight (batch x models) for every approach — shows
    that combining batching + concurrency reaches the GPU's saturation ceiling
    that neither reaches alone."""
    def collect(pred, xfn):
        pts = []
        for c in cfg.values():
            if pred(c):
                pts.append((xfn(c), c["throughput"][0], c["throughput"][1]))
        return sorted(pts)

    lines = [
        ("Batched (1 model)", BLUE,
         collect(lambda c: c["regime"] == "batched", lambda c: c["batch_size"])),
        ("Concurrent, batch 1", AQUA,
         collect(lambda c: c["regime"] == "concurrent" and c["batch_size"] == 1,
                 lambda c: c["n_models"])),
        ("Concurrent + batch 2", YELLOW,
         collect(lambda c: c["regime"] == "concurrent" and c["batch_size"] == 2,
                 lambda c: c["n_models"] * 2)),
        ("Concurrent + batch 4", VIOLET,
         collect(lambda c: c["regime"] == "concurrent" and c["batch_size"] == 4,
                 lambda c: c["n_models"] * 4)),
    ]
    ceiling = max(c["throughput"][0] for c in cfg.values())

    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    _style(ax)
    ax.axhline(ceiling, color=INK2, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate(f"GPU saturation ≈ {ceiling:.0f} img/s", (0.02, ceiling),
                xycoords=("axes fraction", "data"), color=INK2, fontsize=9,
                va="bottom")
    for label, color, pts in lines:
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        ss = [p[2] for p in pts]
        ax.errorbar(xs, ms, yerr=ss, color=color, linewidth=2, marker="o",
                    markersize=6, capsize=3, zorder=3, label=label)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK)
    ax.set_xlabel("images in flight  (batch size × # concurrent models)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("images / second", color=INK2, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title("Reaching the GPU ceiling: batching × concurrency combined\n"
                 "Jetson Orin Nano Super (MAXN_SUPER, MPS)", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    out = os.path.join(FIG, "saturation.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def fig_regime_peak(cfg: dict) -> None:
    """Headline bar chart: best throughput each approach achieves, GPU% annotated."""
    def best(pred):
        cs = [c for c in cfg.values() if pred(c)]
        if not cs:
            return None
        c = max(cs, key=lambda c: c["throughput"][0])
        return c["throughput"][0], c["gpu_util"][0], c["n_models"], c["batch_size"]

    rows = [
        ("Sequential\n(1 model, 1 image)", best(lambda c: c["regime"] == "sequential"), INK2),
        ("Concurrent only\n(6 models × 1)", best(
            lambda c: c["regime"] == "concurrent" and c["batch_size"] == 1), AQUA),
        ("Batched only\n(1 model × 8)", best(lambda c: c["regime"] == "batched"), BLUE),
        ("Concurrent + batched\n(4 models × 4)", best(
            lambda c: c["regime"] == "concurrent" and c["batch_size"] == 4), VIOLET),
    ]
    rows = [r for r in rows if r[1]]
    labels = [r[0] for r in rows]
    vals = [r[1][0] for r in rows]
    utils_pct = [r[1][1] for r in rows]
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for yi, v, u in zip(y, vals, utils_pct):
        ax.annotate(f"{v:.0f} img/s   ·   GPU {u:.0f}%", (v, yi),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("throughput (images / second)", color=INK2, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("One $249 box: peak chest-X-ray throughput by strategy", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left")
    fig.tight_layout()
    out = os.path.join(FIG, "regime_peak.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def fig_trt(cfg: dict) -> None:
    """PyTorch vs TensorRT: the throughput journey on one $249 box (same accuracy)."""
    path = os.path.join(REPO, "results", "trt_bench.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        t = json.load(f)

    rows = [
        ("PyTorch — sequential\n(naive)", t["pytorch_reference_ips"]["sequential"], INK2),
        ("PyTorch — best\n(concurrent + batched)", t["pytorch_reference_ips"]["concurrent_batched_peak"], BLUE),
        ("TensorRT FP16\nsingle stream", t["single_stream_ips"], VIOLET),
        ("TensorRT FP16\nbatched (×8)", t["batched_ips"]["8"], AQUA),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for yi, v in zip(y, vals):
        mult = v / vals[0]
        ax.annotate(f"{v:.0f} img/s   ({mult:.0f}× naive)", (v, yi),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("throughput (images / second)", color=INK2, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title("TensorRT unlocks the box: 25× the naive throughput, same accuracy",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    ax.annotate(f"TensorRT FP16 vs PyTorch FP32: pathology probabilities agree to "
                f"within {t['accuracy_max_pp_diff']} percentage points",
                (0.0, -0.17), xycoords="axes fraction", color=INK2, fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIG, "pytorch_vs_tensorrt.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def fig_tta() -> None:
    """Robustness: single vs TTA vs ensemble AUROC, at ~zero extra latency."""
    path = os.path.join(REPO, "results", "tta_bench.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        t = json.load(f)
    a = t["auroc"]
    rows = [
        ("Single pass", a["single"], INK2),
        (f"TTA ({t['views']} views)", a["tta_5views"], YELLOW),
        (f"Ensemble ({t['ensemble_models']} models)", a["ensemble_3models"], AQUA),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    x = np.arange(len(rows))
    base = a["single"]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    ax.axhline(base, color=INK2, linestyle="--", linewidth=1, zorder=2)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}\n({v-base:+.3f})", (xi, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK, fontsize=10)
    # AUROC differences are small; zoom the y-axis so they're visible.
    lo = min(vals) - 0.01
    ax.set_ylim(lo, max(vals) + 0.012)
    ax.set_ylabel("macro-AUROC over 14 pathologies", color=INK2, fontsize=10)
    ax.set_title("Spending spare capacity on robustness (ChestMNIST, 2000 images)",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    lat = t["latency_ms"]
    ax.annotate(f"cost is ~free: {t['views']} views run as one batch = "
                f"{lat['tta_5views_one_batch']:.0f} ms vs {lat['single']:.0f} ms single "
                f"— ensembling adds +{a['ensemble_3models']-base:.3f} AUROC; naive TTA does not help",
                (0.0, -0.14), xycoords="axes fraction", color=INK2, fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIG, "tta_robustness.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def fig_cost() -> None:
    """Cumulative cost: one-time $249 edge box vs a recurring cloud GPU.

    Illustrative, clearly-caveated (cost is a *hook*, not a hard claim). Assumes a
    clinic-style duty cycle and a comparable cloud GPU hourly rate.
    """
    HW = 249.0                 # Jetson Orin Nano Super one-time
    CLOUD_PER_HR = 0.40        # comparable cloud GPU ($/hr, T4/L4-class)
    HRS_DAY, DAYS_MO = 8, 22   # clinic duty cycle
    ELEC_PER_KWH = 0.30
    EDGE_W = 16.0              # measured concurrent-panel power draw
    months = np.arange(0, 25)

    hrs_mo = HRS_DAY * DAYS_MO
    edge_elec_mo = EDGE_W / 1000 * hrs_mo * ELEC_PER_KWH        # ~ $0.85/mo
    edge = HW + months * edge_elec_mo
    cloud = months * hrs_mo * CLOUD_PER_HR
    breakeven = next((m for m in months if edge[m] <= cloud[m]), None)

    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    _style(ax)
    ax.plot(months, cloud, color=YELLOW, linewidth=2.2, marker="", label="Cloud GPU (recurring)")
    ax.plot(months, edge, color=BLUE, linewidth=2.2, marker="", label="$249 edge box (one-time + power)")
    if breakeven:
        ax.axvline(breakeven, color=INK2, linestyle=":", linewidth=1.2)
        ax.annotate(f"break-even ≈ month {breakeven}", (breakeven, cloud[breakeven]),
                    textcoords="offset points", xytext=(8, -4), color=INK2, fontsize=9)
    ax.set_xlabel("months in service", color=INK2, fontsize=10)
    ax.set_ylabel("cumulative cost (USD)", color=INK2, fontsize=10)
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK)
    ax.set_title("Cost of ownership: one $249 box vs a cloud GPU", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.annotate(f"assumes {HRS_DAY}h/day × {DAYS_MO}d/mo, cloud \\${CLOUD_PER_HR}/h, "
                f"elec \\${ELEC_PER_KWH}/kWh — illustrative",
                (0.0, -0.16), xycoords="axes fraction", color=INK2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "cost_comparison.png")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    recs = _load()
    if not recs:
        print(f"no result JSONs in {RAW}")
        return
    cfg = _by_config(recs)
    print(f"loaded {len(recs)} records across {len(cfg)} configs: {sorted(cfg)}")
    fig_batching(cfg)
    fig_scaling(cfg)
    fig_power_efficiency(cfg)
    fig_saturation(cfg)
    fig_regime_peak(cfg)
    fig_trt(cfg)
    fig_tta()
    fig_cost()


if __name__ == "__main__":
    main()
