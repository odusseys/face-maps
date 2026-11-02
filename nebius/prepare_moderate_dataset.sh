#!/usr/bin/env bash
# Prepare the new dataset only; this script never starts training.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface"
export XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/moderate-prepare.lock"
if ! flock -n 9; then echo 'Moderate preparation is already running.' >&2; exit 1; fi
python="$HOME/miniconda3/envs/faces/bin/python"
source="$FACE_SHARED_ROOT/face_maps/datasets/faces500-moderate-centered"
dataset="$FACE_SHARED_ROOT/face_maps/datasets/faces500-moderate-uv-fixed-texture"
"$python" -u nebius/encode_moderate_portraits.py
"$python" -u nebius/build_uv_aligned_dataset.py --source "$source" --output "$dataset"
"$python" -u encode_shape_latents.py --dataset "$dataset"
"$python" -u nebius/finalize_moderate_dataset.py
