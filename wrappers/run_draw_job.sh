#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag> [fid_profile]"
  exit 1
fi

train_tag=$1
fid_profile=${2:-auto}

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

exec "${repo_dir}/.venv/bin/python" -m workflows.batch_draw_scores --fid-profile "${fid_profile}" "${train_tag}"
