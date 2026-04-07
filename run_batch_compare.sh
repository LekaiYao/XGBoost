#!/bin/bash
set -euo pipefail

echo "===== Batch Compare Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

group_tag=$1
version_start=$2
version_end=$3
output_tag="${group_tag}_v${version_start}_v${version_end}"
echo "Group tag: ${group_tag}"
echo "Version range: v${version_start}-v${version_end}"
echo "Output tag: ${output_tag}"

repo_dir="/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost"
cd "${repo_dir}"

source "${repo_dir}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

if (( version_start > version_end )); then
  echo "Invalid version range: ${version_start} > ${version_end}" >&2
  exit 1
fi

tags=()
for version in $(seq "${version_start}" "${version_end}"); do
  tags+=("${group_tag}_v${version}")
done

echo "Running grouped apply"
python3 batch_apply_scores.py --output-tag "${output_tag}" "${tags[@]}"

echo "Running grouped draw"
python3 batch_draw_scores.py --output-tag "${output_tag}" "${tags[@]}"

echo "===== Batch Compare Job end ====="
echo "Time: $(date)"
