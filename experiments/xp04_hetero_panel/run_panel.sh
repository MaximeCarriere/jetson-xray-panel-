#!/usr/bin/env bash
# XP4 — heterogeneous panel: ResNet-50 (512px) + 3 different-dataset DenseNet-121s,
# concurrent, batch 1. The panel is a *configuration* of XP2's concurrency engine
# (the `--panel` argument), so this just invokes that orchestrator with the right
# model list — there's no separate algorithm to reimplement.
#
#   setsid bash run_panel.sh   (detached; writes ~/conc_run.log)
set -u
cd "$(dirname "$0")"
bash ../xp02_concurrency/run_concurrent.sh --repeats 3 --duration 8 \
    --panel "resnet50-res512-all,densenet121-res224-nih,densenet121-res224-chex,densenet121-res224-pc"
