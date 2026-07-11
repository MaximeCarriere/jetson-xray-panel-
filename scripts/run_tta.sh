#!/usr/bin/env bash
set -u
exec > "$HOME/tta.log" 2>&1
echo "=== run_tta.sh start $(date -Is) args: $* ==="
pkill -9 -f "xray-venv/bin/python" 2>/dev/null
sleep 2
cd "$HOME/jetson-xray-panel/src"
"$HOME/xray-venv/bin/python" tta_experiment.py "$@"
echo "=== run_tta.sh done $(date -Is) ==="
