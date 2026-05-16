#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag>"
  exit 1
fi

train_tag=$1
repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

if [[ "${train_tag}" == *_xgb_* ]]; then
  exec python3 workflows/xgboost_train_direct.py "${train_tag}"
else
  exec python3 workflows/condor_optuna_XGBoost.py "${train_tag}"
fi
