# XP15 — Is your inference time leaking the input? (timing / energy side-channel)

A model that runs offline on an edge box still emits one thing an attacker can measure
without ever seeing the image: **how long each inference takes** (and how much energy it
draws). Timing and power side-channels are a real discipline — SPA/DPA against smartcards,
constant-time cryptography — and they apply to ML inference too. The question this
experiment answers on the actual Jetson: *does the time it takes to classify a chest X-ray
tell you anything about the X-ray?*

> Systems / security study. The interesting result is a **nuance** most edge-AI teams miss,
> not a scare headline.

## What we measured (one image at a time, batch 1 — the real single-query case)

- **A · Content.** Per-pathology-category latency, mean ± std, over the labelled ChestMNIST
  test set — plus **"totally unrelated pictures"** from two angles: degenerate inputs (pure
  noise, all-black, all-white) *and* **real natural photos from a completely different
  domain** (CIFAR-10 cars, cats, ships). If a car classifies in the same time as a chest
  X-ray, content really doesn't leak.
- **B · Model / shape** (positive control). The same measurement across DenseNet@224 vs
  ResNet@512 and two different DenseNet weight sets — to show the channel *is* real
  somewhere.
- **C · Energy.** Sustained per-category power → energy per inference (a single 5–50 ms
  inference is faster than `tegrastats` can sample, so we integrate over a loop).

**Measurement hygiene (this is a side-channel measurement, so it has to be DPA-grade).**
Clocks locked (`jetson_clocks`) and power mode recorded so DVFS can't add input-independent
jitter; warm-up discarded; and — the key control — the categories are measured on a
**shuffled, interleaved schedule** so thermal drift over the run can't line up with a
category and fake a leak. We report **effect size** (microseconds, and as a % of the mean),
not just a p-value: with tens of thousands of samples any trivial difference becomes
"significant", but only a difference an attacker could resolve over a network actually leaks.

## Result — content is invisible, the model is not

![timing side-channel](../../results/figures/timing_sidechannel.png)

**Content — a dead channel.** All 15 pathology categories, the three degenerate inputs
(noise / black / white), **and three classes of real CIFAR photos (car, cat, ship)** land
at **49.55 ms** — a total spread across all 21 groups of **0.066 ms, 0.13 % of the mean**.
A chest X-ray of cardiomegaly (49.54 ms) and a **photo of a car** (49.55 ms) differ by
**11 microseconds**. Energy barely moves either: **~404 mJ per inference**, a ~1 % spread.
Content — medical or not — is simply invisible to the clock.

> The Kruskal-Wallis test reports p ≈ 6 × 10⁻⁹ — technically "significant". That is the
> **p-value trap**: with 18,000 samples even a 0.18 % difference is *detectable*, but an
> **88-microsecond** gap sits far below the millisecond-scale jitter of any real timing
> channel. **Effect size, not the p-value, is what an attacker can exploit — and 0.18 % is
> nothing.** (This is exactly why the experiment reports effect size, not just significance.)

**Model / shape — a live channel.** DenseNet-121 @224 takes **~49.7 ms**; ResNet-50 @512
takes **~24.1 ms** — a **2× difference** an observer resolves instantly. The two DenseNet
weight sets (same architecture) are identical to each other, so what shows is the
**architecture and input resolution**, not the weights or the content.

**Why content doesn't leak, and why this is the *right* answer.** A dense CNN does the
**same arithmetic for every input of a given size** — the convolutions and matmuls have a
fixed operation count regardless of the pixel values, with no data-dependent branching. So
its inference time is *constant-time in the content by construction*. The pathology, and
even a pure-noise image, all take the same time to within measurement noise. That is a
genuinely reassuring security property, and most people never check it.

**What *does* leak** is the part an attacker could actually use: **which model ran** and
**at what input resolution** — different architectures/sizes have visibly different latency.
So if you multiplex several models behind one endpoint, or accept variable-resolution input,
the *timing tells the observer which pipeline they hit* — not what was in the picture.

## The security takeaway

- **Per-image content is not a timing side-channel here** — no constant-time work is needed
  to hide the *image*; the CNN already is.
- **Model identity and input shape are** — the defense that matters is padding every request
  to a **worst-case latency across the models/resolutions you serve**, so the endpoint looks
  identical whichever pipeline runs.
- That padding **costs throughput** (every fast inference waits for the slow worst case), so
  it's a **premium, high-assurance tier** (defense, sovereign, smartcard-adjacent), not the
  default. Feasibility and cost are Pierre's side-channel domain.

*A blog post lurks here: "Your model's inference time is leaking — just not the way you
think."*

## Run (on the board)

```bash
sudo jetson_clocks     # lock clocks first
~/xray-venv/bin/python experiments/xp15_timing_sidechannel/timing_leak.py --per-cat 100 --repeats 10
~/xray-venv/bin/python experiments/xp15_timing_sidechannel/make_figure.py
```
