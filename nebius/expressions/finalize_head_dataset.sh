#!/usr/bin/env bash
# Resume latent preparation and validation; never launch model training.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface" XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/head-dataset-finalization.lock"
if ! flock -n 9; then echo 'Head dataset finalization is already running.' >&2; exit 1; fi
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python="$HOME/miniconda3/envs/faces/bin/python"
"$python" -u -m nebius.expressions.encode_head
"$python" -u -m nebius.expressions.validate_head_dataset
"$python" -u -m nebius.expressions.audit_saved_refined
"$python" - "$FACE_SHARED_ROOT/face_maps/runs/final-refined-audit/results.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f:r=json.load(f)
if len(r)!=8:raise ValueError('Expected eight inverse-render checks')
for x in r:
    if x['shape_error_mm_mean']>.8 or x['expression_error_mm_mean']>.25 or x['expression_error_mm_p95']>1.2 or x['maps_loss_ratio']>.75 or x['condition_loss_ratio']>.75:
        raise ValueError('Saved inverse reconstruction failed quality gate: '+repr(x))
print('DATASET_FINALIZATION_PASSED',flush=True)
PY
