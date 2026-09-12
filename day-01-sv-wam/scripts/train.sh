#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --config config/sv_wam_base.yaml "$@"
