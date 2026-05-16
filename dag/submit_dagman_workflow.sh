#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <group_tag> <version_start> <version_end> [fid_profile]"
  echo "Example: $0 pb24v2_8v_4o200 1 10 fid3"
  exit 1
fi

group_tag=$1
version_start=$2
version_end=$3
fid_profile=${4:-auto}

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

dag_path=$(python3 dag/make_dagman_workflow.py \
  --group-tag "${group_tag}" \
  --version-start "${version_start}" \
  --version-end "${version_end}" \
  --resume-flag 0 \
  --skip-version 0 \
  --draw-only 0 \
  --fid-profile "${fid_profile}" \
  --out-dir "dag/generated" | tail -n 1)

cp dag/submit_templates/submit_staged_single.sub "${afs_dir}/submit_staged_single.sub"
cp dag/submit_templates/submit_batch_compare_single.sub "${afs_dir}/submit_batch_compare_single.sub"
cp wrappers/*.sh "${afs_dir}/"
chmod +x "${afs_dir}"/run_*.sh
mkdir -p "${afs_dir}/$(dirname "${dag_path}")"
cp "${dag_path}" "${afs_dir}/${dag_path}"

echo "DAG generated: ${dag_path}"
echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
