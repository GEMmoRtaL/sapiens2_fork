#!/bin/bash
# Benchmark all Sapiens2 Pose model sizes (0.4B, 0.8B, 1B, 5B).
# Usage:  bash scripts/demo/benchmark_run.sh
#         (from sapiens/pose/ directory)

set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../.." || exit

export SAPIENS_CHECKPOINT_ROOT="/root/Workspace/sapiens2/sapiens2_host"

echo "============================================"
echo " Sapiens2 Pose Speed Benchmark"
echo "============================================"
echo ""

python tools/benchmark_speed.py \
  --det-checkpoint "${SAPIENS_CHECKPOINT_ROOT}/detector/detr-resnet-101-dc5" \
  --checkpoint-root "${SAPIENS_CHECKPOINT_ROOT}/pose" \
  --input ../../demo/data \
  --output /root/Workspace/sapiens2/outputs/benchmark \
  --models 0.4b,0.8b,1b,5b \
  --warmup 3 \
  --measure 100 \
  --device cuda:0 \
  --no-visualize

echo ""
echo "Done. Report: /root/Workspace/sapiens2/outputs/benchmark/benchmark_report.json"
