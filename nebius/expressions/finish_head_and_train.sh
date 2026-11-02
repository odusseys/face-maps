#!/usr/bin/env bash
# Wait for data refinement, validate saved latents, then start requested training.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface" XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
dataset="$FACE_SHARED_ROOT/face_maps/datasets/faces1000-canonical-head-256"
if [[ $# -ne 1 ]]; then echo 'Expected refinement process ID' >&2; exit 2; fi
while [[ ! -s "$dataset/dataset_complete.json" ]]; do
  if ! kill -0 "$1" 2>/dev/null; then echo 'Refinement stopped before completion; training not started.' >&2; exit 1; fi
  sleep 10
done
python="$HOME/miniconda3/envs/faces/bin/python"
"$python" -u -m nebius.expressions.validate_head_dataset
"$python" -u -m nebius.expressions.audit_saved_refined
"$python" - "$FACE_SHARED_ROOT/face_maps/runs/final-refined-audit/results.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert len(r)==8
for x in r:
    if x['shape_error_mm_mean']>.8 or x['expression_error_mm_mean']>.25 or x['expression_error_mm_p95']>1.2 or x['maps_loss_ratio']>.75 or x['condition_loss_ratio']>.75:
        raise ValueError('Saved inverse reconstruction failed quality gate: '+repr(x))
print('SAVED_LATENT_QUALITY_GATE_PASSED',flush=True)
PY
echo 'VALIDATION_COMPLETE_STARTING_TRAINING'
export WANDB_RUN_ID="$(<"$FACE_SHARED_ROOT/face_maps/runs/head-reserved-wandb-id.txt")"
exec bash nebius/expressions/train.sh
