#!/bin/bash
# Convert the final dense P-VQ curriculum checkpoint to compact phase-C state.

set -euo pipefail

: "${PVQ_CURRICULUM_CHECKPOINT:?Set PVQ_CURRICULUM_CHECKPOINT}"

TASK_PYTHON="${SPARSE_EMB_PYTHON:-$(command -v python3.11)}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:-/mnt/local/_outputs/sparse_embedding}"
TASK_INIT_DIR="$TASK_OUTPUT_BASE/compression_init"
TASK_INIT_PATH="$TASK_INIT_DIR/pvq_w768_k128.pt"
test -x "$TASK_PYTHON"
test -d "$PVQ_CURRICULUM_CHECKPOINT"
test ! -e "$TASK_INIT_PATH"
mkdir -p "$TASK_INIT_DIR"

# Recluster the final dense table at K=128. This avoids reusing assignments
# from before the last interval of dense curriculum updates. The converter
# records that scalable capacity-repair k-means replaces the paper's
# vocabulary-quadratic Hungarian balanced assignment.
"$TASK_PYTHON" scripts/convert_dense_baseline.py \
    --method pvq \
    --checkpoint "$PVQ_CURRICULUM_CHECKPOINT" \
    --output "$TASK_INIT_PATH" \
    --pvq_shared_dim 768 \
    --pvq_num_codes 128 \
    --cluster_device cuda \
    --cluster_iters 100 \
    --cluster_restarts 10 \
    --seed 42
