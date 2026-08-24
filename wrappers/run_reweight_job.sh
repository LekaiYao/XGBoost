#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <configured_reweight_tag> [train|apply]" >&2
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
stage=${2:-train}
exec "${repo_dir}/.venv/bin/python" -m workflows.reweighting.run_configured_job "$1" --stage "${stage}"
