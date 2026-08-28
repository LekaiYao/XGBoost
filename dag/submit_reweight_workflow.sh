#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <configured_pp_reweight_tag> [with_splot_validation] [with_mc_domain_validation]" >&2
  exit 1
fi

reweight_tag=$1
with_splot_validation=${2:-1}
with_mc_domain_validation=${3:-1}
for value in "${with_splot_validation}" "${with_mc_domain_validation}"; do
  if [[ ! "${value}" =~ ^[01]$ ]]; then
    echo "ERROR: validation switches must be 0 or 1" >&2
    exit 1
  fi
done

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

generator_output=$("${repo_dir}/.venv/bin/python" -m dag.make_reweight_workflow \
  --reweight-tag "${reweight_tag}" \
  --with-splot-validation "${with_splot_validation}" \
  --with-mc-domain-validation "${with_mc_domain_validation}" \
  --out-dir dag/generated)
echo "${generator_output}"
dag_path=$(echo "${generator_output}" | tail -n 1)

mkdir -p "${afs_dir}/dag/generated" "${afs_dir}/logs"
cp dag/submit_templates/submit_reweight_job.sub "${afs_dir}/submit_reweight_job.sub"
cp dag/submit_templates/submit_reweight_validation.sub "${afs_dir}/submit_reweight_validation.sub"
cp wrappers/run_reweight_job.sh "${afs_dir}/run_reweight_job.sh"
cp wrappers/run_reweight_validation.sh "${afs_dir}/run_reweight_validation.sh"
chmod +x "${afs_dir}/run_reweight_job.sh" "${afs_dir}/run_reweight_validation.sh"
cp "${dag_path}" "${afs_dir}/${dag_path}"

echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
