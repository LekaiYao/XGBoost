#!/bin/bash
set -euo pipefail
train_tag=${1:?Usage: submit_full_apply.sh train_tag}
repo_dir=/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost
afs_dir=/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost
cp "${repo_dir}/wrappers/run_full_apply_job.sh" "${afs_dir}/run_full_apply_job.sh"
chmod +x "${afs_dir}/run_full_apply_job.sh"
cp "${repo_dir}/dag/submit_templates/submit_full_apply_job.sub" "${afs_dir}/submit_full_apply_job.sub"
cd "${afs_dir}"
condor_submit submit_full_apply_job.sub -append "train_tag=${train_tag}" -append "job_tag=${train_tag}_full_apply"
