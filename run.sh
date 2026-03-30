#!/bin/bash

echo "===== Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

train_tag=$1
echo "Train tag: ${train_tag}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"

source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

echo "Python: $(which python3)"

python3 optuna_XGBoost.py ${train_tag}

echo "===== Job end ====="
echo "Time: $(date)"
