#!/bin/bash
set -euo pipefail
cd /eos/home-l/leyao/pbpb_work/X_analysis/XGBoost
export PYTHONNOUSERSITE=1
exec .venv/bin/python -m workflows.punzi_scan "$1"
