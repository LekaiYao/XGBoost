#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag>"
  exit 1
fi

train_tag=$1
repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

if [[ "${train_tag}" == pp* ]]; then
  cmd=(python3 workflow_archive/legacy_non_dag/apply.py "${train_tag}")
else
  cmd=(python3 workflows/batch_apply_scores.py --output-tag "${train_tag}" "${train_tag}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
