#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag> [use_precut] [apply_extra_mc]"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

apply_extra_mc="${3:-0}"
if [[ "${apply_extra_mc}" != "0" && -n "${apply_extra_mc}" ]]; then
  exec "${repo_dir}/.venv/bin/python" -m workflows.batch_apply_scores --apply-extra-mc "${apply_extra_mc}" "$1"
else
  exec "${repo_dir}/.venv/bin/python" -m workflows.batch_apply_scores --use-precut "${2:-0}" "$1"
fi
