# XP9 — Power-envelope sweep

Same TensorRT FP16 batch-8 workload at each of the board's power modes.

## Result
| Mode | Throughput | Power | Efficiency |
|---|---:|---:|---:|
| MAXN_SUPER | 508 img/s | 17.1 W | 29.8 img/s/W |
| **25 W** | 459 img/s | 15.3 W | **30.1 img/s/W** |
| 15 W | 316 img/s | 12.0 W | 26.2 img/s/W |

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
