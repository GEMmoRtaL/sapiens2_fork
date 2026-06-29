#!/bin/bash
# Benchmark inference time vs person count on multi-person images.

set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../.." || exit

export SAPIENS_CHECKPOINT_ROOT="/root/Workspace/sapiens2/sapiens2_host"

echo "============================================"
echo " Multi-Person Inference Time vs Person Count"
echo "============================================"
echo ""

python tools/benchmark_multiperson.py \
  --det-checkpoint "${SAPIENS_CHECKPOINT_ROOT}/detector/detr-resnet-101-dc5" \
  --checkpoint-root "${SAPIENS_CHECKPOINT_ROOT}/pose" \
  --input /root/Workspace/sapiens2/multi-person_dataset \
  --output /root/Workspace/sapiens2/outputs/multiperson \
  --models 0.4b \
  --warmup 2 \
  --repeat 5 \
  --device cuda:0

echo ""
echo "Done. Report: /root/Workspace/sapiens2/outputs/multiperson/multiperson_report.json"
