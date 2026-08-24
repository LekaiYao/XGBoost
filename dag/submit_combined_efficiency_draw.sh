#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <job_tag> <output_tag> <train_tag_a> <train_tag_b>" >&2
  exit 1
fi

job_tag=$1
output_tag=$2
train_tag_a=$3
train_tag_b=$4
repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
"${repo_dir}/.venv/bin/python" -m py_compile workflows/draw_combined_efficiency_mass.py
mkdir -p "${afs_dir}/logs"
cp wrappers/run_combined_efficiency_draw.sh "${afs_dir}/run_combined_efficiency_draw.sh"
cp dag/submit_templates/submit_combined_efficiency_draw.sub "${afs_dir}/submit_combined_efficiency_draw.sub"
chmod +x "${afs_dir}/run_combined_efficiency_draw.sh"

cd "${afs_dir}"
condor_submit submit_combined_efficiency_draw.sub \
  -append "job_tag=${job_tag}" \
  -append "output_tag=${output_tag}" \
  -append "train_tag_a=${train_tag_a}" \
  -append "train_tag_b=${train_tag_b}"
