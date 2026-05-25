#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_tag> [afs_dir]"
  exit 1
fi

train_tag="$1"
afs_dir="${2:-/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost}"

dirs=(
  "$afs_dir/dag/generated"
  "$afs_dir"
)

patterns=(
  "wf_single_${train_tag}.dag.lock"
  "wf_single_${train_tag}.dag.dagman.lock"
  "wf_single_${train_tag}.dag.nodes.log.lock"
  "wf_single_${train_tag}.dag.rescue.lock"
)

removed=0
for d in "${dirs[@]}"; do
  [[ -d "$d" ]] || continue
  for p in "${patterns[@]}"; do
    f="$d/$p"
    if [[ -f "$f" ]]; then
      rm -f "$f"
      echo "removed: $f"
      removed=$((removed + 1))
    fi
  done
done

echo "total_removed_locks: $removed"
