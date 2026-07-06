#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--run" ]]; then
  python3 -m pepp_mlff.training.mace_finetune --config configs/train/mace_finetune.yaml --run
else
  python3 -m pepp_mlff.training.mace_finetune --config configs/train/mace_finetune.yaml
fi
