#!/bin/bash
set -euo pipefail

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${_CONDOR_SCRATCH_DIR:-/tmp}/xgb_mpl_cache"
mkdir -p "${MPLCONFIGDIR}"

exec "${repo_dir}/.venv/bin/python" -m workflows.x_sideband_mass_efficiency "$@"
