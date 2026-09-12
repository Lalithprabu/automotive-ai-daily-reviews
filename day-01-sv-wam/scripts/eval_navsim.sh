#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python evaluate.py --config config/sv_wam_base.yaml "$@"
