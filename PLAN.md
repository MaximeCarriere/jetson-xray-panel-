# Master Plan: Concurrent Multi-Model Chest X-Ray Inference on NVIDIA Jetson Orin Nano Super

> **Purpose of this document.** This is a build spec for a 2-week proof-of-concept.
> It is written to be handed to a coding agent and to a human (Maxime)
> together. It defines the goal, the hardware, the experiments, the code structure,
> the measurement protocol, and the deliverables. Follow it top to bottom.

---

## 0. TL;DR

Measure how concurrent multi-model deep-learning inference scales on a single
NVIDIA Jetson Orin Nano Super (a $249 edge GPU). Use pretrained chest X-ray
pathology models. Produce clean scaling curves (throughput, power, efficiency)
that answer: **"How many disease-detection models can one $300 box run at once,
and how efficiently?"**

This is a **capability + product demo**, NOT a research paper and NOT a medical
product. No claims of clinical validity. No retraining of models. Use pretrained
weights as-is. The value-add is the **systems demonstration** (concurrency,
throughput, power, cost), not model accuracy.

---

## 1. Goal & Non-Goals

### Goal
Demonstrate that a single ~$300 edge device can run a **multi-disease chest X-ray
diagnostic panel** locally (no cloud), and characterize the performance/energy
trade-offs of running multiple models concurrently vs. sequentially vs. batched.

### The audience
Potential clients (clinics, medical-device companies) and internal stakeholders
(Pierre = security/systems cofounder, and an ex-Edge-Impulse advisor). The demo
must make a non-ML buyer think: "That cheap box does a lot — maybe it could do
what we need."

### Non-Goals (explicitly out of scope)
- NOT a clinical product. No regulatory claims. No patient use.
- NOT retraining or improving model accuracy. Use pretrained models as-is.
- NOT the embedded/microcontroller thesis. This is edge-GPU, not MCU. (That's a
  separate track using the XIAO/RISC-V work.)
- NOT the cryptographic security layer (Pierre's part). Mentioned in framing only,
  not built here.
- NOT video / smart-triggering. Static X-ray images only.

---

## 2. Hardware & Environment

- **Device:** NVIDIA Jetson Orin Nano Super, 8GB LPDDR5, 67 TOPS, 25W mode.
- **OS/Runtime:** JetPack 6.x (Ubuntu-based), CUDA, cuDNN, TensorRT preinstalled.
- **Power mode:** Run experiments in the "Super"/25W (MAXN) mode. Record the mode.
- **Power measurement:** `tegrastats` (built-in). No external hardware needed for v1.
- **Storage:** Dataset and models live on SSD/SD card, not RAM. Non-constraint.

### Environment setup checklist
- [ ] Confirm JetPack version (`cat /etc/nv_tegra_release`)
- [ ] Set and lock power mode to MAXN / 25W (`sudo nvpmodel -m 0`, `sudo jetson_clocks`)
- [ ] Install PyTorch build for Jetson (NVIDIA's prebuilt wheel matching JetPack)
- [ ] Install `torchxrayvision`, `torchvision`, `numpy`, `pandas`, `matplotlib`
- [ ] Verify CUDA is visible to PyTorch (`torch.cuda.is_available()`)
- [ ] Install `jetson-stats` (`jtop`) for monitoring
- [ ] Confirm `tegrastats` runs and logs power fields

---

## 3. Dataset & Models

### Dataset
- **Primary:** Kaggle "Chest X-Ray Images (Pneumonia)" (Kermany et al.), ~5,856
  images, ~1.2 GB. Binary normal/pneumonia. Easy, clean.
- **For multi-disease:** use `torchxrayvision`'s bundled test images, or a subset
  of NIH ChestX-ray14 sample images, so we can exercise 14-pathology models.
- Store under `data/`. Do NOT commit images to git (add to `.gitignore`).

### Models (all pretrained, via `torchxrayvision` where possible)
`torchxrayvision` provides DenseNet-121 models pretrained on CheXpert / NIH /
PadChest, each outputting up to 18 pathologies. This is the fastest path.

Models to include in the experiment set:
1. **DenseNet-121 (CheXNet-style)** — ~8M params. The canonical multi-pathology
   model. Outputs 14-18 disease probabilities. This is the workhorse.
2. **DenseNet-121 variants** pretrained on different datasets (CheXpert vs NIH vs
   PadChest) — gives us genuinely *different* models for the concurrent case.
3. **MobileNetV2** (~3.5M params) — smaller, edge-optimized. Comparison point.
4. (Optional) **ResNet-50 / EfficientNet** — larger, for the "accuracy ceiling"
   reference and to stress the GPU harder.

> Note: many "different diseases" are actually different OUTPUT HEADS of one
> multi-label model. Distinguish clearly between:
> - **Multi-label single model**: one DenseNet → 14 disease probabilities (cheap).
> - **Multiple distinct models**: e.g. 4 different DenseNets running concurrently
>   (this is the systems demo — cannot be batched together).

---

## 4. The Core Experiment

**Central question:** How does inference throughput, latency, power, and
efficiency scale as we run more models concurrently on one GPU?

### Three regimes to compare (this comparison IS the result)

1. **Sequential baseline** — N images through 1 model, one at a time. Naive.
2. **Batched** — N images through 1 model as a single batch. The smart
   single-model baseline. (Concurrency must be compared against this, or the
   demo is not credible.)
3. **Concurrent** — N models running truly in parallel (CUDA streams / MPS /
   multiple processes), each handling its own image(s).

Within "concurrent", two sub-cases:
- **3a. Same model ×N** (identical copies) — contends for identical kernels.
- **3b. Different models ×N** — the real diagnostic-panel case; cannot be batched.

### Measurement matrix

| Config | Description | Primary thing it reveals |
|---|---|---|
| S1 | 1 model, 1 image, sequential | baseline latency + power |
| B2/B4/B8 | 1 model, batch size 2/4/8 | batching efficiency |
| C2/C4/C8 (same) | 2/4/8 identical models concurrent | concurrency vs batching |
| C2/C4/C8 (diff) | 2/4/8 different models concurrent | the diagnostic panel scaling |
| RAMP | increase concurrency until saturation | GPU saturation point |

For every config, record:
- **Throughput** (images/sec)
- **Latency** per image (mean, p50, p95)
- **GPU utilization** (%) from tegrastats
- **Power** (W) — average and peak, from tegrastats
- **Memory** (MB) peak
- **Throughput-per-watt** (derived — the efficiency metric)

### Expected shape (hypotheses, to confirm/refute)
- H1: Throughput rises with concurrency, then **plateaus** at GPU saturation.
- H2: Concurrency gain is **sublinear** (4 models ≠ 4× throughput) due to shared
  compute + memory bandwidth.
- H3: For **identical** models, **batching beats concurrency** in throughput.
- H4: For **different** models (can't batch), concurrency is the only option and
  still gives a real gain over sequential.
- H5: There is an **efficiency sweet spot** (best throughput-per-watt) at moderate
  concurrency, not at 1 and not at max.

Confirming or refuting each of these with a graph is the deliverable.

### Stretch experiment: test-time augmentation (TTA) + ensembling

**Idea:** spend the GPU's spare concurrent capacity to improve *diagnostic
robustness*, not just throughput. Instead of one forward pass per image, run N
slightly-augmented views of the same X-ray concurrently and combine their
predictions (average the probabilities, or majority vote).

**Important — frame this correctly.** The value comes from **averaging over
diverse but valid views**, which cancels individual-pass errors. It is NOT
"add noise and hope accuracy goes up" — noise alone usually *hurts*. Use mild,
label-preserving augmentations:
- small rotations (±5-10°)
- mild contrast / brightness shifts
- horizontal flip (valid for chest X-ray in many pathologies — verify per label)
- small center crops / resizes

Two variants to try:
- **TTA on one model:** N augmented views → 1 model → average predictions.
- **Ensemble across different models:** same image → 3 different pretrained
  DenseNets (CheXpert / NIH / PadChest) → average predictions.

**What to measure:**
- Single-pass accuracy (AUROC) vs. TTA/ensemble accuracy — does robustness improve?
- The extra latency + power cost of doing N passes concurrently.
- Net story: "concurrency isn't only throughput — it buys diagnostic robustness
  for modest extra cost, using capacity that would otherwise sit idle."

**Hypothesis H6:** TTA/ensembling via concurrent passes measurably improves
robustness (higher AUROC and/or lower variance on borderline cases) at a modest,
quantifiable latency/power cost.

This is a **stretch / future-work** item. Only pursue after the core three-regime
experiments (Section 4) are complete and clean. If time-limited, describe it as
planned future work rather than running it.

---

## 5. Technical Approach & Gotchas

### Runtime choice
- **Phase 1: PyTorch.** Fast to set up, get results quickly. Start here.
- **Phase 2 (if time): TensorRT.** The real edge-deployment path; better concurrent
  execution and MPS support. Port the best configs only. Do NOT start here — setup
  cost is high.

### Achieving true parallelism (critical gotcha)
- A single GPU has one set of compute units. "Parallel" = the scheduler
  **interleaves** kernels; a second model fills the first's idle gaps. Real gains,
  but sublinear. Set expectations accordingly (this is the finding, not a bug).
- Within ONE Python process, the GIL + single CUDA context limit true concurrency.
  Options, in order of preference for this POC:
  1. **Multiple processes**, one per model, with **CUDA MPS enabled**
     (`nvidia-cuda-mps-control`). This gives genuine concurrent execution.
  2. **CUDA streams** within one process (works, but Python overhead can mask gains).
  3. **Python threading + separate streams** (limited by GIL for CPU-side work).
- Recommendation: implement the concurrent runner with **multiprocessing + MPS**
  as the primary method. Fall back to streams if MPS setup is problematic.

### Warm-up and measurement hygiene
- Always run **warm-up iterations** (discard first ~10) before timing — CUDA
  kernel compilation / caching skews the first runs.
- Time steady-state only. Report mean + p95 over ≥100 inferences.
- Pin power mode and clocks (`jetson_clocks`) so measurements are stable.
- Log tegrastats continuously during each run; align power samples to the run window.
- Run each config **3×** and report mean ± std. Thermal state can drift — note it.

### Memory sanity
- DenseNet-121 FP16 ≈ 16MB weights. Even 10 concurrent ≈ 160MB. 8GB is ample.
  Memory is NOT the constraint; compute + bandwidth are. Don't over-engineer memory.

---

## 6. Repository Structure

```
jetson-xray-panel/
├── README.md                 # what this is, how to run, headline results
├── PLAN.md                   # this document
├── requirements.txt          # or environment notes (Jetson wheels are special)
├── .gitignore                # excludes data/, results/raw/, *.pth
├── data/
│   ├── download.md           # how to get the Kaggle dataset (not committed)
│   └── (images, gitignored)
├── src/
│   ├── models.py             # load pretrained models (torchxrayvision wrappers)
│   ├── power_logger.py       # tegrastats wrapper: start/stop, parse, aggregate
│   ├── runner_sequential.py  # regime 1
│   ├── runner_batched.py     # regime 2
│   ├── runner_concurrent.py  # regime 3 (multiprocessing + MPS)
│   ├── benchmark.py          # orchestrates all configs, writes results
│   └── utils.py              # timing, warmup, image loading, seeding
├── results/
│   ├── raw/                  # per-run JSON (gitignored)
│   └── figures/              # generated graphs (committed)
├── analysis/
│   └── make_figures.py       # reads results/raw, produces the graphs
└── demo/
    └── (later) demo script + 2-min screen recording assets
```

### Data schema (results/raw/*.json)
Each run writes one JSON record:
```json
{
  "config": "C4_diff",
  "regime": "concurrent",
  "n_models": 4,
  "model_names": ["densenet-chexpert", "densenet-nih", ...],
  "batch_size": 1,
  "runtime": "pytorch",
  "power_mode": "MAXN_25W",
  "n_inferences": 200,
  "throughput_ips": 41.2,
  "latency_ms": {"mean": 24.1, "p50": 23.8, "p95": 28.0},
  "gpu_util_pct": 78.5,
  "power_w": {"mean": 18.2, "peak": 20.1},
  "mem_mb_peak": 420,
  "throughput_per_watt": 2.26,
  "repeat_index": 1,
  "timestamp": "..."
}
```

---

## 7. Two-Week Day-by-Day Plan

Assumes near-full-time for 2 weeks. Adjust if part-time (stretch to 3 weeks).

### Week 1 — Setup + single-model + batching

- **Day 1 — Environment.** JetPack check, power mode locked, PyTorch-for-Jetson
  installed, CUDA visible, torchxrayvision working. Download Kaggle dataset.
  Deliverable: `torch.cuda.is_available() == True` and one image classified.
- **Day 2 — Model loading + inference.** `models.py`: load DenseNet + variants +
  MobileNet. Run inference on one image, print pathology probabilities. Confirm
  outputs are sane (match published AUROCs qualitatively).
- **Day 3 — Power logging.** `power_logger.py`: start/stop tegrastats, parse power
  fields, aggregate mean/peak over a window. Validate against `jtop`.
- **Day 4 — Sequential baseline.** `runner_sequential.py` + `benchmark.py` skeleton.
  Measure S1: 1 model, 1 image, warm-up, steady-state timing, power. First JSON
  record written. First clean latency + power number.
- **Day 5 — Batching.** `runner_batched.py`. Measure B2/B4/B8. Produce first graph:
  throughput vs batch size, power vs batch size. This is the single-model story.
- **Day 6 — Buffer / debug.** Fix measurement hygiene issues (warm-up, thermal
  drift, tegrastats alignment). Re-run to confirm stability (3× each, low variance).
- **Day 7 — Checkpoint.** Review Week-1 results. Confirm the single-model baseline
  and batching curves are clean and believable. Write down the numbers.

### Week 2 — Concurrency + panel + graphs

- **Day 8 — Concurrency infra.** Enable CUDA MPS. `runner_concurrent.py` with
  multiprocessing. Get 2 identical models running concurrently, measured. This is
  the hardest engineering day — budget for it.
- **Day 9 — Same-model concurrency.** C2/C4/C8 identical models. Compare to
  batching. Confirm/refute H3 (batching beats concurrency for identical models).
- **Day 10 — Different-model concurrency.** C2/C4/C8 with *different* models (the
  diagnostic panel). This is the money demo. Confirm/refute H4.
- **Day 11 — Saturation ramp.** Increase concurrency until throughput plateaus.
  Find the GPU saturation point and the throughput-per-watt sweet spot (H5).
- **Day 12 — Analysis + figures.** `make_figures.py`: throughput vs concurrency,
  power vs concurrency, throughput-per-watt vs concurrency, batching-vs-concurrency
  comparison. Add a cost line ($300 one-time vs cloud $/month).
- **Day 13 — (Optional) TensorRT port.** Port the best 1-2 configs to TensorRT,
  re-measure. If time-constrained, skip and note as future work.
- **Day 14 — Package.** README with headline results + graphs. Record a 2-minute
  screen demo: the **multi-clinician scenario** (Section 9) — 2-3 simulated
  concurrent clients issuing different model queries against the shared box, with
  live throughput/power readout. Half-page writeup: what it shows, why it matters
  for a clinic.
- **Stretch (if ahead of schedule):** run the TTA/ensemble experiment (Section 4,
  H6) — same image, multiple augmented views / different models, concurrently,
  and measure whether robustness improves for modest extra cost.

---

## 8. Deliverables

1. **Graphs** (the core output):
   - Throughput vs. #concurrent models (sequential / batched / concurrent lines)
   - Power (W) vs. #concurrent models
   - Throughput-per-watt vs. #concurrent models (efficiency sweet spot)
   - Batching vs. concurrency (identical models)
   - Cost comparison: $300 box throughput vs cloud $/month
   - (Stretch) Robustness with vs. without TTA/ensemble, and its latency/power cost
2. **A 2-minute screen-recorded demo**: the multi-clinician scenario — 2-3
   simulated concurrent clients issuing *different* disease queries against one
   shared box, each getting sub-second answers, with live throughput + power
   readout proving parallel execution.
3. **Half-page writeup / short draft**: how multi-model parallel inference works on
   one GPU, what the curves show, and the clinic value story (on-premise, private,
   cheap, no subscription). This is the "small draft" mentioned for later.
4. **Clean repo** with reproducible benchmark scripts.

---

## 9. The Story This Demo Tells (framing for the audience)

- "A single **$300 device** runs a **multi-disease chest X-ray panel** locally."
- "**No cloud** — patient images never leave the building." (privacy / GDPR / HIPAA;
  bridges to Pierre's security work, mentioned not built.)
- "**No subscription** — one-time hardware vs recurring cloud GPU/API cost."
- "We measured exactly **how many models it can run concurrently** and the
  efficiency sweet spot — here are the curves."

### Demo scenario: one box, multiple clinicians, different questions

This is the narrative wrapper around the **concurrent different-models** config
(C2/C4/C8 diff). It is NOT a new experiment — it is the real-world justification
for why concurrent-different-models matters and why batching can't replace it.

Scenario to show in the demo video:
- **Clinician A** queries the box for pneumonia + pleural effusion
- **Clinician B** queries for cardiomegaly + tuberculosis
- **ER doctor C** runs the full 14-pathology panel

...all hitting the same $300 box **simultaneously**, each getting sub-second
answers, no cloud. Because these are *different questions → different models*,
they cannot be batched together — concurrent execution is the only way to serve
them at once. This is exactly the case measured in C*_diff.

Show it as 2-3 simulated concurrent clients in the demo, each issuing different
model queries against the shared device, with the live throughput/power readout
proving they run in parallel.

### Honesty guardrails (do not overclaim)
- Report pretrained models' **published** accuracy; do not claim clinical validity.
- Frame the $300 as a **hook**, not a hard constraint (it also runs on bigger boxes).
- Always show the **batching baseline** so a sharp reviewer can't say "batching
  would be more efficient" — pre-empt it: "yes for one disease; a panel needs
  *different* models, which can't be batched — that's why concurrency matters."
- "True parallel" on a GPU = interleaved kernels filling idle capacity; gains are
  **real but sublinear**. State this plainly; the sublinear curve is the finding.

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MPS / multiprocessing concurrency is fiddly | High | Medium | Budget Day 8 for it; fall back to CUDA streams if MPS fails |
| Thermal throttling skews power/throughput | Medium | Medium | Lock clocks, log temp, run 3×, report drift honestly |
| PyTorch-for-Jetson install pain | Medium | Medium | Use NVIDIA's prebuilt wheel matching exact JetPack version |
| Concurrency gains look unimpressive (very sublinear) | Medium | Low | That's still a valid, honest finding; the *panel* framing (different models can't batch) carries the story regardless |
| Scope creep into accuracy/retraining | Medium | High | Hard rule: pretrained only, no retraining, systems-only focus |
| tegrastats power fields differ across JetPack versions | Low | Low | Parse defensively; validate against jtop |

---

## 11. Instructions for the Coding Agent

When implementing from this spec:

1. **Read this whole file first.** Then confirm the environment (Section 2 checklist)
   before writing benchmark code.
2. **Build in the order of Section 7.** Get the sequential baseline working and
   measured before touching concurrency. Do not jump ahead to MPS.
3. **Every runner writes the JSON schema in Section 6.** Keep results machine-readable
   from day one so `make_figures.py` is trivial later.
4. **Measurement hygiene is mandatory** (Section 5): warm-up, steady-state, 3×
   repeats, continuous power logging aligned to the run window.
5. **Do not retrain models. Do not add accuracy-improvement work.** Pretrained only.
6. **Prefer clarity over cleverness.** This code will be read by a non-specialist
   audience and shown to clients. Comment the concurrency logic especially well.
7. **When something is ambiguous, prefer the simpler measurement** and note the
   assumption in a comment, rather than blocking.
8. Keep `data/` and `results/raw/` out of git. Commit only code, figures, and docs.

---

*Last updated: 2026-05-21. Owner: Maxime Carriere. Hardware provided by Pierre Dubouilh.*
