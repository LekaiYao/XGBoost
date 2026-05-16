#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <train_tag> <optuna_n_trials> <stage_group> <resume_flag> [dataset_year] [selection_profile]"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

exec python3 workflows/train_dispatch.py "$@"
