#!/bin/bash
set -euo pipefail

echo "===== Staged Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

train_tag=$1
optuna_n_trials=$2
stage_group=${3:-}
resume_flag=${4:-0}
dataset_year=${5:-}
selection_profile=${6:-}

if [[ "${dataset_year}" == "__EMPTY__" ]]; then
  dataset_year=""
fi
if [[ "${selection_profile}" == "__EMPTY__" ]]; then
  selection_profile=""
fi

echo "Train tag: ${train_tag}"
echo "OPTUNA_N_TRIALS: ${optuna_n_trials}"
echo "STAGE_GROUP: ${stage_group}"
echo "RESUME: ${resume_flag}"
echo "DATASET_YEAR: ${dataset_year}"
echo "SELECTION_PROFILE: ${selection_profile}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"

source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export TRAINING_MODE=condor
export OPTUNA_N_TRIALS=${optuna_n_trials}

cmd=(python3 staged_optuna_pipeline.py "${train_tag}")
if [[ "${stage_group}" =~ ^v[0-9]+$ || "${stage_group}" =~ ^3v[0-9]+$ ]]; then
  # Use the historical v1-v100 and new 3v1-v100 search-space definitions
  # from condor_optuna_XGBoost.py.
  cmd=(python3 condor_optuna_XGBoost.py "${train_tag}" "${stage_group}")
  if [[ -n "${dataset_year}" ]]; then
    cmd+=(--dataset-year "${dataset_year}")
  fi
  if [[ -n "${selection_profile}" ]]; then
    cmd+=(--selection-profile "${selection_profile}")
  fi
else
  if [[ -n "${stage_group}" ]]; then
    cmd+=(--stage-group "${stage_group}")
  fi
  if [[ "${resume_flag}" == "1" ]]; then
    cmd+=(--resume)
  fi
fi

echo "Python: $(which python3)"
echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "===== Staged Job end ====="
echo "Time: $(date)"
