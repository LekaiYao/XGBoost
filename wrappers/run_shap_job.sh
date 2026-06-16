#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag> [use_precut]"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

use_precut=${2:-0}

exec "${repo_dir}/.venv/bin/python" -m workflows.shap_importance "$1" --use-precut "${use_precut}"
