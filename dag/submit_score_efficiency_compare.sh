#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <comparison_tag> <train_tag_a> <train_tag_b>" >&2
  exit 1
fi

comparison_tag=$1
train_tag_a=$2
train_tag_b=$3
repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
"${repo_dir}/.venv/bin/python" -m py_compile workflows/compare_score_efficiency.py
mkdir -p "${afs_dir}/logs"
cp wrappers/run_score_efficiency_compare.sh "${afs_dir}/run_score_efficiency_compare.sh"
cp dag/submit_templates/submit_score_efficiency_compare.sub "${afs_dir}/submit_score_efficiency_compare.sub"
chmod +x "${afs_dir}/run_score_efficiency_compare.sh"

cd "${afs_dir}"
condor_submit submit_score_efficiency_compare.sub \
  -append "comparison_tag=${comparison_tag}" \
  -append "train_tag_a=${train_tag_a}" \
  -append "train_tag_b=${train_tag_b}"
