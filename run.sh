#!/bin/bash

echo "===== Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

train_tag=$1
optuna_n_trials=$2
echo "Train tag: ${train_tag}"
echo "OPTUNA_N_TRIALS: ${optuna_n_trials}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"

source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export OPTUNA_N_TRIALS=${optuna_n_trials}

echo "Python: $(which python3)"

python3 optuna_XGBoost.py ${train_tag}

echo "===== Job end ====="
echo "Time: $(date)"
