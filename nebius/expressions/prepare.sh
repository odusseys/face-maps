#!/usr/bin/env bash
# Resume dataset preparation and validation only; this never invokes training.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface" XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/expression-prepare.lock"
if ! flock -n 9; then echo 'Expression preparation is already running.' >&2; exit 1; fi
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python="$HOME/miniconda3/envs/faces/bin/python"
"$python" -u nebius/expressions/render.py --workers 8
"$python" -u -m nebius.expressions.build
"$python" -u -m nebius.expressions.encode
"$python" -u -m nebius.expressions.validate
CUDA_VISIBLE_DEVICES= "$python" -u -m nebius.expressions.prompt
