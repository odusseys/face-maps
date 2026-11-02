#!/usr/bin/env bash
# Prepare and validate diverse paired data; this script does not start training.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface"
export XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/diverse-prepare.lock"
if ! flock -n 9; then echo 'Diverse preparation is already running.' >&2; exit 1; fi
python="$HOME/miniconda3/envs/faces/bin/python"
source="$FACE_SHARED_ROOT/face_maps/datasets/faces1000-visibleface-centered"
dataset="$FACE_SHARED_ROOT/face_maps/datasets/faces1000-visibleface-uv-paired"
"$python" -u nebius/encode_diverse_portraits.py
"$python" -u nebius/build_diverse_uv_dataset.py --source "$source" --output "$dataset"
"$python" -u nebius/encode_diverse_targets.py --dataset "$dataset"
"$python" -u nebius/validate_paired_dataset.py --dataset "$dataset"
"$python" -u nebius/audit_diverse_latents.py
