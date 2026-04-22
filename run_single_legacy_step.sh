#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <train|apply|draw|shap> <train_tag>"
  exit 1
fi

step=$1
train_tag=$2

echo "===== Single Legacy Step start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "Step: ${step}"
echo "Train tag: ${train_tag}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
legacy_dir="${repo_dir}/workflow_archive/legacy_non_dag"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

case "${step}" in
  train)
    cmd=(python3 "${legacy_dir}/XGBoost.py" "${train_tag}")
    ;;
  apply)
    cmd=(python3 "${legacy_dir}/apply.py" "${train_tag}")
    ;;
  draw)
    cmd=(python3 "${legacy_dir}/draw.py" "${train_tag}")
    ;;
  shap)
    cmd=(python3 "${repo_dir}/shap_importance.py" "${train_tag}")
    ;;
  *)
    echo "Unknown step: ${step}. Expected one of: train, apply, draw, shap" >&2
    exit 1
    ;;
esac

echo "Python: $(which python3)"
echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "===== Single Legacy Step end ====="
echo "Time: $(date)"
