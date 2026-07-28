#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag> [with_shap] [use_precut]"
  echo "Example: $0 X_pb24_v2_fid1_8v1_xgb_v1 0 0"
  echo "Example: $0 Bd_pb24_v1_fid1_6v1_xgb_v1 0 1"
  exit 1
fi

train_tag=$1
with_shap=${2:-0}
use_precut=${3:-0}

if [[ ! "${train_tag}" =~ ^(X|Bu|Bd|Bs)_(pp24|pb23|pb24)_v[0-9]+_fid[0-9]+_[0-9]+v[0-9]*(_rw[a-z0-9]+)?_xgb_v[0-9]+$ ]]; then
  echo "ERROR: invalid single DAG train_tag '${train_tag}'"
  echo "Expected format: {channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}"
  echo "Example: X_pb24_v2_fid1_8v1_xgb_v1"
  exit 1
fi

if [[ ! "${use_precut}" =~ ^[01]$ ]]; then
  echo "ERROR: use_precut must be 0 or 1"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
python_bin="${repo_dir}/.venv/bin/python"

if [[ "${use_precut}" == "1" ]]; then
  if [[ ! "${train_tag}" =~ ^Bd_(pb23|pb24)_ ]]; then
    echo "ERROR: use_precut=1 only supports Bd_pb23_* and Bd_pb24_* single-DAG tags"
    exit 1
  fi
  "${python_bin}" -m workflows.prepare_bd_pbpb_precut_inputs "${train_tag}" --input-dir "input"
fi

dag_path=$("${python_bin}" -m dag.make_single_workflow --train-tag "${train_tag}" --with-shap "${with_shap}" --use-precut "${use_precut}" --out-dir "dag/generated")

cp dag/submit_templates/submit_train_job.sub "${afs_dir}/submit_train_job.sub"
cp dag/submit_templates/submit_apply_job.sub "${afs_dir}/submit_apply_job.sub"
cp dag/submit_templates/submit_draw_job.sub "${afs_dir}/submit_draw_job.sub"
cp dag/submit_templates/submit_shap_job.sub "${afs_dir}/submit_shap_job.sub"
cp wrappers/*.sh "${afs_dir}/"
chmod +x "${afs_dir}"/run_*.sh
mkdir -p "${afs_dir}/dag/generated"
cp "${dag_path}" "${afs_dir}/${dag_path}"

echo "DAG generated: ${dag_path}"
echo "Dry-run train submit template"
(
  cd "${afs_dir}" && \
  condor_submit -dry-run /tmp/train_job.dry submit_train_job.sub \
    -append "train_tag=${train_tag}" \
    -append "use_precut=${use_precut}" \
    -append "job_tag=dryrun_train_job"
)

echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
