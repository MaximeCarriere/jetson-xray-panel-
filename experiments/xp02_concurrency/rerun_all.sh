#!/usr/bin/env bash
# QA re-run of the full concurrency matrix (XP2 + XP3 + XP4) in one consistent session.
set -u
exec > "$HOME/rerun_conc.log" 2>&1
echo "=== rerun_all start $(date -Is) ==="
echo a | sudo -S jetson_clocks 2>/dev/null   # ensure clocks locked
pkill -9 -f "xray-venv/bin/python" 2>/dev/null; echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
cd "$HOME/jetson-xray-panel/experiments/xp02_concurrency"
P="$HOME/xray-venv/bin/python"
echo ">>> XP2 concurrency (same/diff/ramp)"
$P benchmark_concurrent.py --repeats 3 --duration 8 --same 2,4,5 --diff 2,4,5 --ramp 3,6,7
echo ">>> XP3 concurrent+batched (batch 4)"
$P benchmark_concurrent.py --repeats 3 --duration 8 --same 2,3,4 --diff 2,4 --ramp , --batch 4
echo ">>> XP3 concurrent+batched (batch 2)"
$P benchmark_concurrent.py --repeats 3 --duration 8 --same 2,4,6 --diff , --ramp , --batch 2
echo ">>> XP4 heterogeneous panel"
$P benchmark_concurrent.py --repeats 3 --duration 8 --panel "resnet50-res512-all,densenet121-res224-nih,densenet121-res224-chex,densenet121-res224-pc"
echo "=== rerun_all done $(date -Is) ==="
