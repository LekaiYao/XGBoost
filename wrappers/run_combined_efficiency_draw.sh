#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <output_tag> <train_tag_a> <train_tag_b>" >&2
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${_CONDOR_SCRATCH_DIR:-/tmp}/xgb_mpl_cache"
mkdir -p "${MPLCONFIGDIR}"
exec "${repo_dir}/.venv/bin/python" -m workflows.draw_combined_efficiency_mass "$1" "$2" "$3"
