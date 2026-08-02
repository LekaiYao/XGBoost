#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <replica_index> [output_dir]"
  exit 1
fi

replica_index=$1
output_dir=${2:-/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/output/reweighting/diagnostics_pp24_psi2s_bootstrap_v1}
repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
exec "${repo_dir}/.venv/bin/python" -m workflows.reweighting.bootstrap_diagnostics \
  --replicas-only \
  --replicas 1 \
  --replica-start "${replica_index}" \
  --skip-existing \
  --output-dir "${output_dir}"
