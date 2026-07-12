# XP12 — Energy accounting + load-adaptive power governor

Can we spend *less energy* serving the panel by scaling the board's power envelope
to the live load? We measure Joules/image and put a control loop on top of the XP11
server that moves between 15 W / 25 W / MAXN as demand changes.

**Governor policy** (`governor.py`): measure the request rate each second; **scale up
immediately** when demand rises (defend the SLA), **scale down lazily** (hysteresis +
5 s dwell) to avoid flapping. Thresholds from the measured SLA-safe capacities
(XP9/XP11): 15 W ≈ 300, 25 W ≈ 460, MAXN ≈ 510 req/s.

**Evaluation** (`governor_eval.py`): replay a bursty clinic-day profile (quiet ↔
rounds ↔ peaks of 440–450 req/s) under three policies, integrating VDD_IN → energy.

## Result — an honest tradeoff, not a free lunch

| Policy | Energy/img | Avg power | p99 | SLA violations |
|---|---:|---:|---:|---:|
| always-MAXN | 49.9 mJ | 11.5 W | 55 ms | **0.0 %** |
| always-25 W | 47.5 mJ (−4.7 %) | 10.9 W | 334 ms | 21.9 % |
| **adaptive** | 48.2 mJ (−3.4 %) | 11.1 W | 202 ms | 9.1 % |

![governor](../../results/figures/governor.png)

The governor **works** — the timeline shows it dropping to 15 W in quiet periods and
jumping to MAXN for each burst — and it lands on a real point of the energy↔SLA
frontier: **better SLA than static-25 W (9 % vs 22 % miss) at less energy than
always-MAXN (−3.4 %)**.

### But read the honest conclusions:
1. **The energy savings from power-mode scaling are modest (~3–5 %)**, not dramatic.
   At low load the power draw *converges* across modes — the work is the same and the
   GPU races to idle — so the power cap barely matters. The lever mostly bites at
   sustained high load.
2. **15 W is actually the *least* energy-efficient mode per image** (XP9: 26 vs 30
   img/s/W) — lower peak power ≠ lower energy. It's for peak-power/thermal caps, not
   for saving energy.
3. **The real energy lever is utilisation, not power mode** — keep the GPU busy
   (batch), and it's ~30 img/s/W regardless. Power mode is a few-percent trim.
4. **Reactive lag costs SLA:** the governor takes ~1–2 s to escalate, so it can't
   fully match MAXN's 0 % (a predictive or shorter-dwell policy would help).

So for a strictly latency-critical service you'd just run MAXN; the governor earns
its keep when energy/thermal budget matters more than the last few percent of tail
SLA. Reporting this honestly is the point.

## Run
```bash
~/xray-venv/bin/python governor_eval.py     # runs all three policies, restores MAXN
```

## Files
`governor.py` (adaptive controller + `set_mode`) · `governor_eval.py` (3-policy eval).
Energy integration in `lib/power_logger.py` (`energy_joules`). Builds on XP11's
`serving.py`. Data `../../results/governor_bench.json`.
