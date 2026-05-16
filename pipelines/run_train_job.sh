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

sample="pbpb"
if [[ "${train_tag}" == pp* ]]; then
  sample="pp"
fi

if [[ "${train_tag}" =~ _o([0-9]+)_v[0-9]+$ ]]; then
  export OPTUNA_N_TRIALS="${BASH_REMATCH[1]}"
fi

if [[ "${sample}" == "pp" ]]; then
  echo "[TRAIN] sample=pp, train_tag=${train_tag}"
  cmd=(python3 workflow_archive/legacy_non_dag/XGBoost.py "${train_tag}")
else
  echo "[TRAIN] sample=pbpb, train_tag=${train_tag}, trials=${OPTUNA_N_TRIALS:-default}"
  cmd=(python3 workflows/condor_optuna_XGBoost.py "${train_tag}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
