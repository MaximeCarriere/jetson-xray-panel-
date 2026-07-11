# Live multi-clinician demo

A terminal dashboard that runs the **different-models panel** live: several
clinicians hit one $249 Jetson at once, each asking a different question →
different model → which can't be batched, so they run concurrently. The readout
shows each clinician's throughput plus the shared box's GPU utilisation and power,
proving they run in parallel on one GPU with no cloud.

## Run

On the board (models download on first run; CUDA MPS is started automatically):

```bash
~/xray-venv/bin/python ~/jetson-xray-panel/demo/demo_panel.py --seconds 25
```

Options: `--clients N` (1–4), `--seconds T`, `--no-mps`.

## Record

For the 2-minute demo video:

1. SSH into the board in a **full-screen terminal** (dark theme reads best).
2. Pre-warm once (`--seconds 3`) so the recording has no weight-download pause.
3. Start the screen recording, then run `demo_panel.py --seconds 25`.
4. Narrate the three points as the numbers settle:
   - four **different** disease models, one shared GPU, **no cloud**;
   - **GPU ~95–99% utilised, ~18 W** — the whole panel on a 25 W box;
   - different questions **can't be batched** → concurrency is why this works.

## What it shows (measured)

| Clinician | Model | Throughput | Latency |
|---|---|---:|---:|
| Screening | DenseNet-121 (NIH) | ~15 img/s | 66 ms |
| Cardiology | DenseNet-121 (CheXpert) | ~15 img/s | 66 ms |
| ER full panel | DenseNet-121 (all) | ~15 img/s | 66 ms |
| Radiologist (512px) | ResNet-50 | ~30 img/s | 33 ms |
| **Aggregate** | 4 models concurrent | **~75 img/s** | GPU ~97% |
