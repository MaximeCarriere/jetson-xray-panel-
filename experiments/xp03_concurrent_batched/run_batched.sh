#!/usr/bin/env bash
# XP3 — concurrent + batched: N model copies, each running a batch of B. This is a
# *configuration* of XP2's concurrency engine (the `--batch` argument), not a new
# algorithm — so this just invokes that orchestrator at batch 4 and batch 2. There's
# no core runner of its own for the same reason as XP4.
#
#   setsid bash run_batched.sh   (detached; writes ~/conc_run.log)
set -u
cd "$(dirname "$0")"
ORCH=../xp02_concurrency/run_concurrent.sh
bash "$ORCH" --repeats 3 --duration 8 --same 2,3,4 --ramp , --batch 4
bash "$ORCH" --repeats 3 --duration 8 --same 2,4,6 --ramp , --batch 2
