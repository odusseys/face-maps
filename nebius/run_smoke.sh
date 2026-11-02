#!/usr/bin/env bash
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface"
export XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
export WANDB_API_KEY="$(<"$FACE_SHARED_ROOT/credentials/wandb_api_key")"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=odusseys WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/smoke.lock"
if ! flock -n 9; then echo 'Face smoke test is already running.' >&2; exit 1; fi
cd "$FACE_SHARED_ROOT/code/face_pipeline"
exec "$HOME/miniconda3/envs/faces/bin/python" -u nebius/smoke_train.py \
  --dataset "$FACE_SHARED_ROOT/face_maps/datasets/smoke-6-similarity" \
  --output "$FACE_SHARED_ROOT/face_maps/training/nebius-smoke" \
  --displacement-only --epochs 3 --steps 3 --batch-size 4 --micro-batch-size 1 \
  --log-every 3 --checkpoint-every 3 --inference-steps 28 \
  --wandb-project flux2-face-displacement-only --wandb-name nebius-smoke
