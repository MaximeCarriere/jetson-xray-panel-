# XP10 — Sustained-load thermal endurance

10 minutes of continuous TensorRT FP16 (batch-8) inference — does the throughput hold, or
does the board **throttle** once it heats up?

## What is thermal throttling, and why test for it?

Silicon has a temperature limit. When a chip's junction temperature approaches its ceiling
(~87 °C on Orin), the hardware protects itself by **automatically lowering its own clock
speed** — "thermal throttling." Lower clocks = lower throughput. Crucially, this is
invisible in a short benchmark: a 10-second burst runs cool and posts a great number, but
the same workload run for real, for minutes, can heat-soak and quietly lose 20–40% of that
speed once throttling kicks in.

So every peak number in this repo (508 img/s) raises a fair question: **is it a real,
sustainable rate, or just a cold-start burst you could never deploy?** This experiment
answers it — run the workload continuously and watch temperature *and* throughput together.

## Result
- **Throughput: 507 → 508 img/s (−0.2 %)** — dead flat for the full 10 minutes (the
  −0.2 % is measurement noise, not decay; re-verified in a second run).
- **Temperature rises from ~61 °C and plateaus at ~69 °C** (max 69.1 °C) — it reaches
  **thermal equilibrium** (heat generated = heat dissipated) with a ~18 °C margin below the
  ~87 °C throttle threshold, and stops climbing.
- **Steady 18 W, GPU pinned at 98 %** the entire time.

**No thermal throttling — the box holds full throughput indefinitely.** The 508 img/s
headline is a number you can actually run 24/7, not a benchmark artifact. That's the
difference between a demo and a deployable always-on device.

![endurance](../../results/figures/endurance.png)

### The honest caveat
The board *can* show **transient** throttling — but only from many heavy runs fired
back-to-back with **no cooldown**, where residual heat accumulates across runs. From a
**cool start under a single sustained load** (the realistic deployment case), it stays in
equilibrium at ~69 °C and never throttles. Note this is a *thermal* result specifically:
it says the chip doesn't overheat, which is separate from the board's power-delivery
stability under spiky load.

## Run
```bash
~/xray-venv/bin/python endurance.py ~/densenet_nih_fp16.engine --minutes 10
```

## Files
`endurance.py`. Data `../../results/endurance.json`.
