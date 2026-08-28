#!/bin/bash
set -euo pipefail

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
cp dag/submit_templates/submit_x_sideband_mass_efficiency.sub "${afs_dir}/submit_x_sideband_mass_efficiency.sub"
cp wrappers/run_x_sideband_mass_efficiency.sh "${afs_dir}/run_x_sideband_mass_efficiency.sh"
chmod +x "${afs_dir}/run_x_sideband_mass_efficiency.sh"
mkdir -p "${afs_dir}/logs"

cd "${afs_dir}"
condor_submit submit_x_sideband_mass_efficiency.sub
