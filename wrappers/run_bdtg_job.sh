#!/bin/bash
set -euo pipefail
cd "${BDTG_REPO_DIR:-/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
exec .venv/bin/python -m workflows.bdtg_workflow "$@"
