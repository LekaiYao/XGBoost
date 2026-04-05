#!/bin/bash
set -euo pipefail


echo "===== Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

train_tag=$1
optuna_n_trials=$2
search_space_tag=$3
echo "Train tag: ${train_tag}"
echo "OPTUNA_N_TRIALS: ${optuna_n_trials}"
echo "SEARCH_SPACE_TAG: ${search_space_tag}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"

source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export TRAINING_MODE=condor
export OPTUNA_N_TRIALS=${optuna_n_trials}

echo "Python: $(which python3)"

python3 condor_optuna_XGBoost.py ${train_tag} ${search_space_tag}

echo "===== Job end ====="
echo "Time: $(date)"
