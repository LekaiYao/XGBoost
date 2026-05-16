#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from configs.samples import infer_dataset_year, infer_selection_profile
from utils.varsets import VARSETS, infer_sample_from_tag, infer_varset_from_tag


def train_tag(group_tag: str, version: int) -> str:
    return f"{group_tag}_v{version}"


def parse_optuna_n_trials_from_group_tag(group_tag: str) -> int:
    matches = re.findall(r"(?:^|_)(?:\d+o|o)(\d+)(?:_|$)", group_tag)
    if not matches:
        raise ValueError(
            f"Cannot infer optuna_n_trials from group_tag '{group_tag}'. "
            "Expected token like '_o200' or '_4o200'."
        )
    return int(matches[-1])


def validate_group_varset(group_tag: str) -> str:
    sample = infer_sample_from_tag(group_tag)
    varset = infer_varset_from_tag(group_tag, sample=sample)
    if varset is None:
        raise ValueError(
            f"Unable to infer <varset> from group_tag '{group_tag}'. "
            f"Expected one of: {sorted(VARSETS.get(sample, {}).keys())}"
        )
    return sample, varset


def make_dag(
    out_dir: Path,
    group_tag: str,
    version_start: int,
    version_end: int,
    optuna_n_trials: int,
    resume_flag: int,
    skip_version: int,
    draw_only: int,
    dataset_year: str,
    selection_profile: str,
    fid_profile: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_path = out_dir / f"wf_{group_tag}_v{version_start}_v{version_end}.dag"
    dataset_year_var = dataset_year if dataset_year else "__EMPTY__"
    selection_profile_var = selection_profile if selection_profile else "__EMPTY__"

    lines = []
    for version in range(version_start, version_end + 1):
        node = f"TR_v{version}"
        tag = train_tag(group_tag, version)
        stage_group = f"v{version}"
        job_tag = f"{tag}_staged"
        lines.append(f"JOB {node} submit_staged_single.sub")
        lines.append(
            f'VARS {node} train_tag="{tag}" stage_group="{stage_group}" '
            f'optuna_n_trials="{optuna_n_trials}" resume_flag="{resume_flag}" job_tag="{job_tag}" '
            f'dataset_year="{dataset_year_var}" selection_profile="{selection_profile_var}"'
        )
        lines.append("")

    post_node = "POST_BATCH"
    post_job_tag = f"{group_tag}_v{version_start}_v{version_end}_batchcmp"
    data_input_override_var = "__EMPTY__"
    output_prefix_var = "__EMPTY__"
    lines.append(f"FINAL {post_node} submit_batch_compare_single.sub")
    lines.append(
        f'VARS {post_node} group_tag="{group_tag}" version_start="{version_start}" '
        f'version_end="{version_end}" skip_version="{skip_version}" draw_only="{draw_only}" '
        f'version_token="v" dataset_year="{dataset_year_var}" '
        f'data_input_override="{data_input_override_var}" output_prefix="{output_prefix_var}" '
        f'fid_profile="{fid_profile}" job_tag="{post_job_tag}"'
    )
    lines.append("")

    dag_path.write_text("\n".join(lines))
    return dag_path


def main():
    parser = argparse.ArgumentParser(description="Generate one DAG: parallel training + final batch apply/draw.")
    parser.add_argument("--group-tag", required=True)
    parser.add_argument("--version-start", type=int, required=True)
    parser.add_argument("--version-end", type=int, required=True)
    parser.add_argument("--resume-flag", type=int, default=0)
    parser.add_argument("--skip-version", type=int, default=0)
    parser.add_argument("--draw-only", type=int, default=0)
    parser.add_argument("--fid-profile", default="auto")
    parser.add_argument("--out-dir", default="dag/generated")
    args = parser.parse_args()

    if args.version_start > args.version_end:
        raise ValueError("version-start must be <= version-end")

    sample, varset = validate_group_varset(args.group_tag)
    dataset_year = infer_dataset_year(args.group_tag, sample)
    selection_profile = infer_selection_profile(args.group_tag, sample)
    optuna_n_trials = parse_optuna_n_trials_from_group_tag(args.group_tag)
    print(f"Detected varset: {varset}")
    print(f"Detected sample: {sample}")
    print(f"Varset columns: {VARSETS[sample][varset]}")
    print(f"Detected dataset_year: {dataset_year}")
    print(f"Detected selection_profile: {selection_profile}")
    print(f"Detected optuna_n_trials: {optuna_n_trials}")

    dag = make_dag(
        out_dir=Path(args.out_dir),
        group_tag=args.group_tag,
        version_start=args.version_start,
        version_end=args.version_end,
        optuna_n_trials=optuna_n_trials,
        resume_flag=args.resume_flag,
        skip_version=args.skip_version,
        draw_only=args.draw_only,
        dataset_year=dataset_year,
        selection_profile=selection_profile,
        fid_profile=args.fid_profile,
    )
    print(dag)


if __name__ == "__main__":
    main()
