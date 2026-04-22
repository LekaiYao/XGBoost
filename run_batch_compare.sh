#!/bin/bash
set -euo pipefail

echo "===== Batch Compare Job start ====="
echo "Host: $(hostname)"
echo "Time: $(date)"

group_tag=$1
version_start=$2
version_end=$3
skip_version=${4:-0}
draw_only=${5:-0}
version_token=${6:-v}
dataset_year=""
data_input_override=""
output_prefix=""
fid_profile="auto"

if [[ $# -ge 10 ]]; then
  dataset_year=${7:-}
  data_input_override=${8:-}
  output_prefix=${9:-}
  fid_profile=${10:-auto}
else
  data_input_override=${7:-}
  output_prefix=${8:-}
  fid_profile=${9:-auto}
fi

if [[ "${dataset_year}" == "__EMPTY__" ]]; then
  dataset_year=""
fi
if [[ "${data_input_override}" == "__EMPTY__" ]]; then
  data_input_override=""
fi
if [[ "${output_prefix}" == "__EMPTY__" ]]; then
  output_prefix=""
fi
output_tag="${group_tag}_${version_token}${version_start}_${version_token}${version_end}"
echo "Group tag: ${group_tag}"
echo "Version range: ${version_token}${version_start}-${version_token}${version_end}"
echo "Skip version: ${skip_version}"
echo "Draw only: ${draw_only}"
echo "Version token: ${version_token}"
echo "Dataset year: ${dataset_year}"
echo "Data input override: ${data_input_override}"
echo "Output prefix: ${output_prefix}"
echo "FID profile: ${fid_profile}"
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
  if [[ "${version}" == "${skip_version}" ]]; then
    continue
  fi
  tags+=("${group_tag}_${version_token}${version}")
done

if [[ "${draw_only}" == "1" ]]; then
  echo "Skipping grouped apply (draw_only=1)"
else
  echo "Running grouped apply"
  apply_cmd=(python3 batch_apply_scores.py --output-tag "${output_tag}")
  if [[ -n "${dataset_year}" ]]; then
    apply_cmd+=(--dataset-year "${dataset_year}")
  fi
  if [[ -n "${data_input_override}" ]]; then
    apply_cmd+=(--data-input "${data_input_override}")
  fi
  if [[ -n "${output_prefix}" ]]; then
    apply_cmd+=(--output-prefix "${output_prefix}")
  fi
  apply_cmd+=("${tags[@]}")
  "${apply_cmd[@]}"
fi

echo "Running grouped draw"
draw_cmd=(python3 batch_draw_scores.py --output-tag "${output_tag}")
if [[ -n "${output_prefix}" ]]; then
  draw_cmd+=(--output-prefix "${output_prefix}")
fi
if [[ -n "${fid_profile}" ]]; then
  draw_cmd+=(--fid-profile "${fid_profile}")
fi
draw_cmd+=("${tags[@]}")
"${draw_cmd[@]}"

echo "===== Batch Compare Job end ====="
echo "Time: $(date)"
