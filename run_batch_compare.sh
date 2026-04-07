#!/bin/bash
set -euo pipefail

echo "===== Batch Compare Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

group_tag=$1
echo "Group tag: ${group_tag}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"

source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

tags=(
  "${group_tag}_v1"
  "${group_tag}_v2"
  "${group_tag}_v3"
  "${group_tag}_v4"
  "${group_tag}_v5"
  "${group_tag}_v6"
  "${group_tag}_v7"
  "${group_tag}_v8"
  "${group_tag}_v9"
  "${group_tag}_v10"
)

echo "Running grouped apply"
python3 batch_apply_scores.py "${tags[@]}"

echo "Running grouped draw"
python3 batch_draw_scores.py "${tags[@]}"

echo "===== Batch Compare Job end ====="
echo "Time: $(date)"
