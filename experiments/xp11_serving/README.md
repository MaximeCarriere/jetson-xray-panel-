# XP11 — Dynamic-batching serving layer + latency under load

## The one-line point

Every other experiment measured **throughput** — how many images/second the engine can
crunch if you hand it a big pile of them at once (XP6: 510 img/s). But a real deployment
isn't a pile — it's a **stream of requests arriving one at a time, at random moments**,
and each one needs an *answer back quickly*. This experiment asks the question that
throughput can't: **how many requests per second can the box actually serve while still
answering each one fast enough?** The answer (≈482 req/s) is *lower* than the raw
throughput (510 img/s), and this shows exactly why.

## The vocabulary (so the table makes sense)

- **Request / RPS.** One X-ray coming in to be classified. **RPS = requests per second** =
  the *offered load*, how fast they arrive.
- **Latency.** Time from a request arriving to its answer being ready — queue wait +
  waiting to be batched + inference. This is what a user feels.
- **p50 / p95 / p99 (percentiles).** Latency isn't one number, it's a distribution, so we
  report percentiles. **p50** = the median (half the requests are faster). **p99** = the
  *slow tail* — 99% of requests are at least this fast, 1% are slower. Tail latency
  matters because the 1% of patients who wait 3 seconds are the ones who complain.
- **SLA (Service-Level Agreement).** The promise you must keep. Here: **p99 < 100 ms** —
  "99% of reads come back in under a tenth of a second." A load level "passes" only if it
  keeps that promise.

## How the server works

- **The server** (`serving.py`): each model has a **request queue** and a **worker thread**
  that owns its TensorRT context + CUDA stream. The worker grabs waiting requests, forms a
  **batch** when it either has `max_batch` (8) of them **or** `max_delay_ms` (5 ms) has
  passed — whichever comes first — runs the engine once on the whole batch, and answers
  them together. This is **dynamic batching**: at light load it runs tiny batches (low
  latency); as load rises it naturally forms bigger batches (high throughput). It adapts
  on its own — measured mean batch size climbs from ~1 to 8 (mean 5.7).
- **The load generator** (`loadgen.py`): fires requests at a target RPS with **open-loop
  Poisson arrivals** — random spacing, like real independent clients, and "fire-and-forget"
  so a slow response doesn't hold up the next arrival (a closed loop would hide the
  problem). It sweeps the RPS and records the latency percentiles at each level.

## Result — the "queueing hockey stick"

DenseNet-121 FP16, `max_batch=8`, `max_delay=5 ms`, SLA = **p99 < 100 ms**.

| Offered RPS | p50 | p95 | p99 | SLA |
|---:|---:|---:|---:|:--:|
| 100 | 11 ms | 17 ms | 20 ms | ✅ |
| 300 | 19 ms | 29 ms | 33 ms | ✅ |
| 450 | 26 ms | 40 ms | 47 ms | ✅ |
| **500** | 44 ms | 87 ms | **99 ms** | ✅ (the knee) |
| 550 | 475 ms | 651 ms | 673 ms | ❌ |
| 700 | 2055 ms | 3332 ms | 3458 ms | ❌ |

![serving latency](../../results/figures/serving_latency.png)

**Read the shape, not just the rows.** Latency stays *flat and low* (tens of ms) all the
way up to ~500 RPS, then **explodes** — 99 ms at 500 RPS becomes 673 ms at 550 and 3458 ms
at 700. On the log-scale plot that sudden turn upward is the **"hockey stick."** The bend
is called the **knee**, and it happens at the **capacity** (≈482 req/s here).

## Why it explodes — the intuition

Think of a checkout line. The server processes requests at some maximum **service rate**
(its capacity). While requests **arrive slower than that**, any little pile-up gets worked
off between bursts, so the queue stays short and latency stays low. The moment requests
**arrive faster than the server can clear them**, the queue has nowhere to drain — it grows
and grows, and every new request waits behind an ever-longer line. Latency doesn't rise
gently; it runs away. (This is classic **M/M/1 queueing** behaviour — a standing result in
queueing theory.)

## The lesson a throughput number hides

The raw engine does **510 img/s**, but the **SLA-safe serving capacity is only ~482 req/s**.
You **cannot** run a queue at 100% utilisation: the closer offered load gets to capacity,
the more the tail latency blows up, so you must leave headroom. "510 img/s" is the speed
limit; "482 req/s" is the speed you can actually *drive at* without crashing the SLA — and
the second number is the one that matters for deployment.

## Run
```bash
~/xray-venv/bin/python loadgen.py --rps 25,50,100,200,300,400,450,500,550,600,700 \
    --duration 8 --sla-p99-ms 100
```

## Files
`serving.py` (DynamicBatcher + Server) · `loadgen.py` (open-loop Poisson load).
Runtime `lib/trt_runner.py`. Data `../../results/serving_bench.json`.

## Next (XP12)
This server is the substrate for the **energy-adaptive governor**: a control loop that
scales the board's power envelope (15 W ↔ 25 W ↔ MAXN) to the live load to minimise
Joules/image while holding this SLA.
