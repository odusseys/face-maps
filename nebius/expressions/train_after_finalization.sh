#!/usr/bin/env bash
# Start training only after the active data-only finalizer passes every check.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd "$FACE_SHARED_ROOT/code/face_pipeline"
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 8>"$FACE_SHARED_ROOT/face_maps/runs/head-training-after-finalization.lock"
if ! flock -n 8; then echo 'A post-finalization launch is already queued.' >&2; exit 1; fi
if [[ $# -ne 1 ]]; then echo 'Expected active data-finalizer PID.' >&2; exit 2; fi
log="$FACE_SHARED_ROOT/face_maps/runs/head-finalize-data-only.log"
dataset="$FACE_SHARED_ROOT/face_maps/datasets/faces1000-canonical-head-256"
echo "WAITING_FOR_DATA_FINALIZATION pid=$1"
while ! grep -Fqx 'DATASET_FINALIZATION_PASSED' "$log"; do
  if ! kill -0 "$1" 2>/dev/null; then echo 'Dataset finalizer exited without passing; training not started.' >&2; exit 1; fi
  sleep 10
done
for file in "$dataset/dataset_complete.json" "$dataset/validation.json" "$FACE_SHARED_ROOT/face_maps/runs/final-refined-audit/results.json" "$FACE_SHARED_ROOT/face_maps/eval/reference-portrait/aligned.png" "$FACE_SHARED_ROOT/face_maps/runs/head-reserved-wandb-id.txt"; do
  if [[ ! -s "$file" ]]; then echo "Required finalization artifact missing: $file" >&2; exit 1; fi
done
export WANDB_RUN_ID="$(<"$FACE_SHARED_ROOT/face_maps/runs/head-reserved-wandb-id.txt")"
echo 'DATASET_CHECKS_PASSED_STARTING_TRAINING'
exec bash nebius/expressions/train.sh
