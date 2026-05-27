#!/bin/bash
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--dry-run] <train_tag> [afs_dir]"
  exit 1
fi

train_tag="$1"
afs_dir="${2:-/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost}"
dag_gen_dir="$afs_dir/dag/generated"

patterns=(
  "wf_single_${train_tag}.dag.lock"
  "wf_single_${train_tag}.dag.dagman.lock"
  "wf_single_${train_tag}.dag.nodes.log.lock"
  "wf_single_${train_tag}.dag.rescue.lock"
  "wf_single_${train_tag}.dag.condor.sub"
  "wf_single_${train_tag}.dag.lib.out"
  "wf_single_${train_tag}.dag.lib.err"
  "wf_single_${train_tag}.dag.dagman.log"
  "wf_single_${train_tag}.dag.dagman.out"
  "wf_single_${train_tag}.dag.metrics"
  "wf_single_${train_tag}.dag.nodes.log"
)

removed=0
for p in "${patterns[@]}"; do
  f="$dag_gen_dir/$p"
  if [[ -f "$f" ]]; then
    if [[ $dry_run -eq 1 ]]; then
      echo "would_remove: $f"
    else
      rm -f "$f"
      echo "removed: $f"
    fi
    removed=$((removed + 1))
  fi
done

if [[ $dry_run -eq 1 ]]; then
  echo "total_would_remove: $removed"
else
  echo "total_removed: $removed"
fi
