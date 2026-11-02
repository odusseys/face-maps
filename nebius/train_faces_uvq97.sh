#!/usr/bin/env bash
# Single-texture displacement training, with resumable checkpoints and Forge W&B.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface"
export XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
wandb_key_file="$FACE_SHARED_ROOT/credentials/wandb_api_key"
if [[ ! -s "$wandb_key_file" ]]; then echo 'CoreWeave Forge W&B credential is missing.' >&2; exit 1; fi
export WANDB_API_KEY="$(<"$wandb_key_file")"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export WANDB_ENTITY="${WANDB_ENTITY:-odusseys}"
export WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/fixed-texture-training.lock"
if ! flock -n 9; then echo 'Fixed-texture training is already running.' >&2; exit 1; fi
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/miniconda3/envs/faces/bin/python" -u nebius/train_fixed_texture.py \
  --dataset "$FACE_SHARED_ROOT/face_maps/datasets/faces500-moderate-uv-fixed-texture" \
  --output "$FACE_SHARED_ROOT/face_maps/training/flux2_moderate_fixed_texture" \
  --displacement-only --epochs 100 --steps 10000 --resume-latest --log-every 200 --checkpoint-every 200 \
  --wandb-project flux2-face-displacement-only --wandb-name nebius-moderate-fixed-texture-10k "$@"
