#!/bin/bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "Usage: $0 <group_tag> <version_start> <version_end> <skip_version> <draw_only> <version_token> <dataset_year> <data_input_override> <output_prefix> <fid_profile>"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

exec python3 -m workflows.group_apply_draw "$@"
