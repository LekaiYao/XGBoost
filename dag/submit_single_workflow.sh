#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag> [with_shap]"
  echo "Example: $0 X_pb24_v2_fid1_8v_xgb_v1 0"
  exit 1
fi

train_tag=$1
with_shap=${2:-0}

if [[ ! "${train_tag}" =~ ^(X|Bu|Bd|Bs)_(pp24|pb23|pb24)_v[0-9]+_fid[0-9]+_[0-9]+v[0-9]*_xgb_v[0-9]+$ ]]; then
  echo "ERROR: invalid single DAG train_tag '${train_tag}'"
  echo "Expected format: {channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}"
  echo "Example: X_pb24_v2_fid1_8v_xgb_v1"
  exit 1
fi

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
python_bin="${repo_dir}/.venv/bin/python"

dag_path=$("${python_bin}" -m dag.make_single_workflow --train-tag "${train_tag}" --with-shap "${with_shap}" --out-dir "dag/generated")

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
    -append "job_tag=dryrun_train_job"
)

echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
