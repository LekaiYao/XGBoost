#!/usr/bin/env python3
import argparse
from pathlib import Path

from configs.samples import infer_dataset_year, infer_selection_profile
from utils.tagging import parse_optuna_spec_from_group_body, split_channel_tag
from utils.varsets import get_varset_columns, infer_channel_from_tag, infer_sample_from_tag, infer_varset_from_tag


def train_tag(group_tag: str, version: int) -> str:
    return f"{group_tag}_v{version}"


def parse_optuna_spec_from_group_tag(group_tag: str):
    _, body = split_channel_tag(group_tag)
    return parse_optuna_spec_from_group_body(body)


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
        train_node = f"TR_v{version}"
        apply_node = f"AP_v{version}"
        draw_node = f"DR_v{version}"
        tag = train_tag(group_tag, version)
        train_job_tag = f"{tag}_train"
        apply_job_tag = f"{tag}_apply"
        draw_job_tag = f"{tag}_draw"

        lines.append(f"JOB {train_node} submit_train_dispatch_single.sub")
        lines.append(
            f'VARS {train_node} train_tag="{tag}" '
            f'optuna_n_trials="{optuna_n_trials}" job_tag="{train_job_tag}" '
            f'dataset_year="{dataset_year_var}" selection_profile="{selection_profile_var}"'
        )
        lines.append("")

        lines.append(f"JOB {apply_node} submit_apply_job.sub")
        lines.append(f'VARS {apply_node} train_tag="{tag}" job_tag="{apply_job_tag}"')
        lines.append("")

        lines.append(f"JOB {draw_node} submit_draw_job.sub")
        lines.append(
            f'VARS {draw_node} train_tag="{tag}" fid_profile="{fid_profile}" job_tag="{draw_job_tag}"'
        )
        lines.append("")

        lines.append(f"PARENT {train_node} CHILD {apply_node}")
        lines.append(f"PARENT {apply_node} CHILD {draw_node}")
        lines.append("")

    dag_path.write_text("\n".join(lines))
    return dag_path


def main():
    parser = argparse.ArgumentParser(description="Generate one DAG: parallel training + final batch apply/draw.")
    parser.add_argument("--group-tag", required=True)
    parser.add_argument("--version-start", type=int, required=True)
    parser.add_argument("--version-end", type=int, required=True)
    parser.add_argument("--fid-profile", default="auto")
    parser.add_argument("--out-dir", default="dag/generated")
    args = parser.parse_args()

    if args.version_start > args.version_end:
        raise ValueError("version-start must be <= version-end")

    sample, varset = validate_group_varset(args.group_tag)
    dataset_year = infer_dataset_year(args.group_tag, sample)
    selection_profile = infer_selection_profile(args.group_tag, sample)
    optuna_objective_index, optuna_n_trials, optuna_space_version = parse_optuna_spec_from_group_tag(args.group_tag)
    print(f"Detected varset: {varset}")
    print(f"Detected sample: {sample}")
    print(f"Varset columns: {get_varset_columns(sample, varset, channel=infer_channel_from_tag(args.group_tag))}")
    print(f"Detected dataset_year: {dataset_year}")
    print(f"Detected selection_profile: {selection_profile}")
    print(f"Detected optuna_objective_index: {optuna_objective_index}")
    print(f"Detected optuna_n_trials: {optuna_n_trials}")
    print(f"Detected optuna_space_version: {optuna_space_version}")

    dag = make_dag(
        out_dir=Path(args.out_dir),
        group_tag=args.group_tag,
        version_start=args.version_start,
        version_end=args.version_end,
        optuna_n_trials=optuna_n_trials,
        dataset_year=dataset_year,
        selection_profile=selection_profile,
        fid_profile=args.fid_profile,
    )
    print(dag)


if __name__ == "__main__":
    main()
