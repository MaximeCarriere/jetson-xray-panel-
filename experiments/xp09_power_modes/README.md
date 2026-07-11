# XP9 — Power-envelope sweep

Same TensorRT FP16 batch-8 workload at each of the board's power modes.

## Result
Mean ± SE over 3 runs per mode.

| Mode | Throughput | Power | Efficiency |
|---|---:|---:|---:|
| MAXN_SUPER | 508.5 ± 0.6 img/s | 17.2 W | 29.58 ± 0.12 img/s/W |
| **25 W** | 459.0 ± 0.6 img/s | 15.4 W | **29.86 ± 0.09 img/s/W** |
| 15 W | 316.3 ± 0.0 img/s | 12.1 W | 26.07 ± 0.05 img/s/W |

**MAXN for peak throughput; 25 W for best efficiency** (≈90% of the throughput,
1.8 W less). 15 W throttles too hard. For a battery/fanless clinic box, 25 W is the
operating point.

![power modes](../../results/figures/power_modes.png)

## Run
```bash
~/xray-venv/bin/python power_sweep.py ~/densenet_nih_fp16.engine
```

## Files
`power_sweep.py` (sets nvpmodel, restores MAXN_SUPER after). Data
`../../results/power_sweep.json`.
