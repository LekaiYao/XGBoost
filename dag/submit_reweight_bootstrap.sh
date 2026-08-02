#!/bin/bash
set -euo pipefail

replica_count=${1:-100}
if [[ ! "${replica_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: replica_count must be a positive integer"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"
output_dir="${repo_dir}/output/reweighting/diagnostics_pp24_psi2s_bootstrap_v1"

cd "${repo_dir}"
mkdir -p "${output_dir}/replicas" "${afs_dir}/logs"
cp wrappers/run_reweight_bootstrap_replica.sh "${afs_dir}/run_reweight_bootstrap_replica.sh"
cp dag/submit_templates/submit_reweight_bootstrap_replica.sub "${afs_dir}/submit_reweight_bootstrap_replica.sub"
chmod +x "${afs_dir}/run_reweight_bootstrap_replica.sh"

cd "${afs_dir}"
condor_submit submit_reweight_bootstrap_replica.sub \
  -append "replica_count=${replica_count}" \
  -append "output_dir=${output_dir}"
