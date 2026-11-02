#!/usr/bin/env bash
# Regenerate and validate canonical-expression data; never starts training.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface" XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/canonical-prepare.lock"
if ! flock -n 9; then echo 'Canonical preparation is already running.' >&2; exit 1; fi
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python="$HOME/miniconda3/envs/faces/bin/python"
dataset="$FACE_SHARED_ROOT/face_maps/datasets/faces1000-canonical-expression-256"
"$python" -u -m nebius.expressions.render_canonical
"$python" -u -m nebius.expressions.build_canonical
"$python" -u -m nebius.expressions.encode --dataset "$dataset"
"$python" -u -m nebius.expressions.validate_canonical
