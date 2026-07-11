#!/usr/bin/env bash
# Run the concurrent benchmark with a clean environment. Args are passed through
# to benchmark_concurrent.py. Designed to be launched detached:
#   setsid bash ~/jetson-xray-panel/scripts/run_concurrent.sh <args> &
set -u
LOG="$HOME/conc_run.log"
exec > "$LOG" 2>&1
echo "=== run_concurrent.sh start $(date -Is) args: $* ==="
pkill -9 -f "xray-venv/bin/python" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null
sleep 2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
cd "$HOME/jetson-xray-panel/src"
"$HOME/xray-venv/bin/python" benchmark_concurrent.py "$@"
echo "=== run_concurrent.sh done $(date -Is) ==="
