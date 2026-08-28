#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <reweight_tag> <splot|ppref-pbpb> <apply_job|none>" >&2
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${_CONDOR_SCRATCH_DIR:-/tmp}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if [[ "$2" == "splot" ]]; then
  exec "${repo_dir}/.venv/bin/python" -m workflows.reweighting.validate_configured_reweight "$1" splot
fi

exec "${repo_dir}/.venv/bin/python" -m workflows.reweighting.validate_configured_reweight \
  "$1" ppref-pbpb --apply-job "$3"
