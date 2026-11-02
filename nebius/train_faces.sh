#!/usr/bin/env bash
# Default face experiment: diverse paired texture and displacement training.
set -euo pipefail
exec bash "$(dirname "$0")/train_diverse_paired.sh" "$@"
