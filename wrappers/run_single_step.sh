#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <step> <train_tag>"
  exit 1
fi

step=$1
train_tag=$2
repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python_bin="${repo_dir}/.venv/bin/python"

case "${step}" in
  train)
    if [[ "${train_tag}" == *_xgb_* ]]; then
      exec "${python_bin}" -m workflows.xgboost_train_direct "${train_tag}"
    else
      exec "${python_bin}" -m workflows.condor_optuna_XGBoost "${train_tag}"
    fi
    ;;
  apply)
    exec "${python_bin}" -m workflows.batch_apply_scores "${train_tag}"
    ;;
  draw)
    exec "${python_bin}" -m workflows.batch_draw_scores "${train_tag}"
    ;;
  shap)
    exec "${python_bin}" -m workflows.shap_importance "${train_tag}"
    ;;
  *)
    echo "Unknown step: ${step}" >&2
    exit 2
    ;;
esac
