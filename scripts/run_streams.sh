#!/usr/bin/env bash
# Detached launcher for the CUDA-streams scaling sweep. Args -> runner_streams.py.
set -u
exec > "$HOME/streams.log" 2>&1
echo "=== run_streams.sh start $(date -Is) args: $* ==="
pkill -9 -f "xray-venv/bin/python" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null
sleep 2
cd "$HOME/jetson-xray-panel/src"
"$HOME/xray-venv/bin/python" runner_streams.py "$@"
echo "=== run_streams.sh done $(date -Is) ==="
