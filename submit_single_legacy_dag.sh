#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag>"
  echo "Example: $0 pp24_5v_xgb_v1"
  exit 1
fi

train_tag=$1

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
afs_dir="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost"

cd "${repo_dir}"
source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

dag_path=$(python3 make_single_legacy_dag.py --train-tag "${train_tag}" --out-dir "dags")

cp submit_single_legacy_step.sub "${afs_dir}/submit_single_legacy_step.sub"
cp run_single_legacy_step.sh "${afs_dir}/run_single_legacy_step.sh"
mkdir -p "${afs_dir}/dags"
cp "${dag_path}" "${afs_dir}/${dag_path}"

echo "DAG generated: ${dag_path}"
echo "Dry-run single-step submit template"
(
  cd "${afs_dir}" && \
  condor_submit -dry-run /tmp/single_legacy_step.dry submit_single_legacy_step.sub \
    -append "step=train" \
    -append "train_tag=${train_tag}" \
    -append "job_tag=dryrun_single_legacy_step"
)

echo "Submitting DAG from AFS: ${dag_path}"
(
  cd "${afs_dir}" && \
  condor_submit_dag "${dag_path}"
)
