#!/usr/bin/env bash
# Resumable face-data preparation. Run on the Nebius VM, not on the cluster.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface"
export XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/prepare-faces.lock"
if ! flock -n 9; then
  echo 'Face dataset preparation is already running.' >&2
  exit 1
fi
python="$HOME/miniconda3/envs/faces/bin/python"
code="$FACE_SHARED_ROOT/code/face_pipeline"
source_data="$FACE_SHARED_ROOT/face_maps/datasets/faces500-centered"
dataset="$FACE_SHARED_ROOT/face_maps/datasets/faces500-similarity-linear-q97"
cd "$code"
"$python" -u generate_faces.py --count 500 --holdout 100 --output "$source_data" \
  --render-workers "${FACE_RENDER_WORKERS:-2}" --render-threads "${FACE_RENDER_THREADS:-4}"
"$python" -u remap_similarity_q97.py --source "$source_data" --output "$dataset"
"$python" -u encode_shape_latents.py --dataset "$dataset"
"$python" -u cache_face_prompts.py
echo FACE_DATA_READY
