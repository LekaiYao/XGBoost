#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <group_tag> <version_start> <version_end> <version_token> [optuna_n_trials] [dataset_year] [selection_profile] [fid_profile]"
  echo "Example: $0 pb24v2_8v_4o200 1 10 v 200 2024 pb24v2 fid3"
  exit 1
fi

group_tag=$1
version_start=$2
version_end=$3
version_token=$4
optuna_n_trials=${5:-200}
dataset_year=${6:-}
selection_profile=${7:-}
fid_profile=${8:-auto}
dataset_year_arg=${dataset_year}
selection_profile_arg=${selection_profile}
if [[ -z "${dataset_year_arg}" ]]; then
  dataset_year_arg="__EMPTY__"
fi
if [[ -z "${selection_profile_arg}" ]]; then
  selection_profile_arg="__EMPTY__"
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

dag_path=$(python3 dag/make_dagman_workflow.py \
  --group-tag "${group_tag}" \
  --version-start "${version_start}" \
  --version-end "${version_end}" \
  --version-token "${version_token}" \
  --optuna-n-trials "${optuna_n_trials}" \
  --resume-flag 0 \
  --skip-version 0 \
  --draw-only 0 \
  --dataset-year "${dataset_year_arg}" \
  --selection-profile "${selection_profile_arg}" \
  --fid-profile "${fid_profile}" \
  --out-dir "dag/generated" | tail -n 1)

cp dag/submit_templates/submit_staged_single.sub "${afs_dir}/submit_staged_single.sub"
cp dag/submit_templates/submit_batch_compare_single.sub "${afs_dir}/submit_batch_compare_single.sub"
cp pipelines/run_staged.sh "${afs_dir}/run_staged.sh"
cp pipelines/run_batch_compare.sh "${afs_dir}/run_batch_compare.sh"
chmod +x "${afs_dir}/run_staged.sh" "${afs_dir}/run_batch_compare.sh"
mkdir -p "${afs_dir}/$(dirname "${dag_path}")"
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
    -append "dataset_year=${dataset_year_arg}" \
    -append "selection_profile=${selection_profile_arg}" \
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
    -append "dataset_year=${dataset_year_arg}" \
    -append "data_input_override=__EMPTY__" \
    -append "output_prefix=__EMPTY__" \
    -append "fid_profile=${fid_profile}" \
    -append "job_tag=dryrun_batch"
)

echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
