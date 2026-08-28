#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage (explicit mode): $0 <group_tag_with_{n}o{N}_v{k}> [fid_profile] [with_shap] [with_fit_interface]"
  echo "Example: $0 X_pb24_v2_fid2_18v1_1o200_v1 auto 1 1"
  echo "Usage (legacy batch):  $0 <group_tag_with_{n}o{N}> <version_start> <version_end> [fid_profile] [with_shap] [with_fit_interface]"
  echo "Example: $0 X_pb24_v2_fid2_18v1_1o200 1 10 auto 1 1"
  exit 1
fi

group_tag=$1
version_start=""
version_end=""
fid_profile="auto"
with_shap=1
with_fit_interface=1

if [[ $# -ge 3 ]]; then
  version_start=$2
  version_end=$3
  fid_profile=${4:-auto}
  with_shap=${5:-1}
  with_fit_interface=${6:-1}
else
  fid_profile=${2:-auto}
  with_shap=${3:-1}
  with_fit_interface=${4:-1}
fi

if [[ ! "${with_shap}" =~ ^[01]$ ]]; then
  echo "ERROR: with_shap must be 0 or 1"
  exit 1
fi

if [[ ! "${with_fit_interface}" =~ ^[01]$ ]]; then
  echo "ERROR: with_fit_interface must be 0 or 1"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
python_bin="${repo_dir}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

if [[ -n "${version_start}" && -n "${version_end}" ]]; then
  dag_path=$("${python_bin}" -m dag.make_dagman_workflow \
    --group-tag "${group_tag}" \
    --version-start "${version_start}" \
    --version-end "${version_end}" \
    --fid-profile "${fid_profile}" \
    --with-shap "${with_shap}" \
    --with-fit-interface "${with_fit_interface}" \
    --out-dir "dag/generated" | tail -n 1)
else
  dag_path=$("${python_bin}" -m dag.make_dagman_workflow \
    --group-tag "${group_tag}" \
    --fid-profile "${fid_profile}" \
    --with-shap "${with_shap}" \
    --with-fit-interface "${with_fit_interface}" \
    --out-dir "dag/generated" | tail -n 1)
fi

cp dag/submit_templates/submit_train_dispatch_single.sub "${afs_dir}/submit_train_dispatch_single.sub"
cp dag/submit_templates/submit_apply_job.sub "${afs_dir}/submit_apply_job.sub"
cp dag/submit_templates/submit_draw_job.sub "${afs_dir}/submit_draw_job.sub"
cp dag/submit_templates/submit_shap_job.sub "${afs_dir}/submit_shap_job.sub"
cp dag/submit_templates/submit_fit_interface_job.sub "${afs_dir}/submit_fit_interface_job.sub"
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
