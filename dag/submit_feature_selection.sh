#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment>"
  exit 1
fi
repo=/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost
afs=/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost
cd "$repo"
export PYTHONNOUSERSITE=1
dag_path=$(.venv/bin/python -m dag.make_feature_selection_workflow "$1")
mkdir -p "$afs/dag/generated" "$afs/logs"
for step in train apply draw shap feature_selection_rank; do
  if [[ "$step" == feature_selection_rank ]]; then
    cp "dag/submit_templates/submit_${step}.sub" "$afs/"
    cp "wrappers/run_${step}.sh" "$afs/"
  else
    cp "dag/submit_templates/submit_${step}_job.sub" "$afs/"
    cp "wrappers/run_${step}_job.sh" "$afs/"
  fi
done
cp "$dag_path" "$afs/$dag_path"
cd "$afs"
condor_submit_dag "$dag_path"
