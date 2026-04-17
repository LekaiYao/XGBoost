#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <group_tag> <version_start> <version_end> <version_token> [optuna_n_trials]"
  echo "Example: $0 pb23v5_4v2_4o200 11 20 2v 200"
  exit 1
fi

group_tag=$1
version_start=$2
version_end=$3
version_token=$4
optuna_n_trials=${5:-200}

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

dag_path=$(python3 make_dagman_workflow.py \
  --group-tag "${group_tag}" \
  --version-start "${version_start}" \
  --version-end "${version_end}" \
  --version-token "${version_token}" \
  --optuna-n-trials "${optuna_n_trials}" \
  --resume-flag 0 \
  --skip-version 0 \
  --draw-only 0 \
  --out-dir "dags")

cp submit_staged_single.sub "${afs_dir}/submit_staged_single.sub"
cp submit_batch_compare_single.sub "${afs_dir}/submit_batch_compare_single.sub"
mkdir -p "${afs_dir}/dags"
cp "${dag_path}" "${afs_dir}/${dag_path}"

echo "DAG generated: ${dag_path}"
echo "Dry-run staged submit template"
(
  cd "${afs_dir}" && \
  condor_submit -dry-run /tmp/staged_single.dry submit_staged_single.sub \
    -append "train_tag=${group_tag}_${version_token}${version_start}" \
    -append "stage_group=${version_token}${version_start}" \
    -append "optuna_n_trials=${optuna_n_trials}" \
    -append "resume_flag=0" \
    -append "job_tag=dryrun_staged"
)

echo "Dry-run batch submit template"
(
  cd "${afs_dir}" && \
  condor_submit -dry-run /tmp/batch_single.dry submit_batch_compare_single.sub \
    -append "group_tag=${group_tag}" \
    -append "version_start=${version_start}" \
    -append "version_end=${version_end}" \
    -append "skip_version=0" \
    -append "draw_only=0" \
    -append "version_token=${version_token}" \
    -append "job_tag=dryrun_batch"
)

echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
