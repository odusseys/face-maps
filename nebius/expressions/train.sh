#!/usr/bin/env bash
# Explicit training entrypoint. Dataset preparation never invokes this file.
set -euo pipefail
export FACE_SHARED_ROOT="${FACE_SHARED_ROOT:-$HOME/datasets/faces}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export HF_HOME="$FACE_SHARED_ROOT/face_maps/cache/huggingface" XDG_CACHE_HOME="$FACE_SHARED_ROOT/face_maps/cache"
key_file="$FACE_SHARED_ROOT/credentials/wandb_api_key"
if [[ ! -s "$key_file" ]]; then echo 'CoreWeave Forge W&B credential is missing.' >&2; exit 1; fi
export WANDB_API_KEY="$(<"$key_file")" WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export WANDB_ENTITY="${WANDB_ENTITY:-odusseys}" WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
mkdir -p "$FACE_SHARED_ROOT/face_maps/runs"
exec 9>"$FACE_SHARED_ROOT/face_maps/runs/shape-expression-training.lock"
if ! flock -n 9; then echo 'Expression training is already running.' >&2; exit 1; fi
cd "$FACE_SHARED_ROOT/code/face_pipeline"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/miniconda3/envs/faces/bin/python" -u -m nebius.expressions.train --resume-latest "$@"
