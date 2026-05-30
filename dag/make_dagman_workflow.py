#!/usr/bin/env python3
import argparse
from pathlib import Path

from utils.tagging import parse_optuna_spec_from_group_body, split_channel_tag
from utils.varsets import infer_sample_from_tag


def train_tag(group_tag: str, version: int, mode: str) -> str:
    if mode == "explicit":
        return group_tag
    return f"{group_tag}_v{version}"


def parse_optuna_spec_from_group_tag(group_tag: str):
    _, body = split_channel_tag(group_tag)
    return parse_optuna_spec_from_group_body(body)


def make_dag(
    out_dir: Path,
    group_tag: str,
    mode: str,
    version_start: int,
    version_end: int,
    optuna_n_trials: int,
    fid_profile: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_path = out_dir / f"wf_{group_tag}_v{version_start}_v{version_end}.dag"

    lines = []
    for version in range(version_start, version_end + 1):
        train_node = f"TR_v{version}"
        apply_node = f"AP_v{version}"
        draw_node = f"DR_v{version}"
        tag = train_tag(group_tag, version, mode)
        train_job_tag = f"{tag}_train"
        apply_job_tag = f"{tag}_apply"
        draw_job_tag = f"{tag}_draw"

        lines.append(f"JOB {train_node} submit_train_dispatch_single.sub")
        lines.append(
            f'VARS {train_node} train_tag="{tag}" '
            f'optuna_n_trials="{optuna_n_trials}" job_tag="{train_job_tag}"'
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
    parser.add_argument("--version-start", type=int, default=None)
    parser.add_argument("--version-end", type=int, default=None)
    parser.add_argument("--fid-profile", default="auto")
    parser.add_argument("--out-dir", default="dag/generated")
    args = parser.parse_args()

    sample = infer_sample_from_tag(args.group_tag)
    mode, optuna_objective_index, optuna_n_trials, optuna_space_version = parse_optuna_spec_from_group_tag(args.group_tag)
    if mode == "explicit":
        if args.version_start is not None or args.version_end is not None:
            raise ValueError(
                "Explicit mode '{n}o{N}_v{k}' does not accept version-start/version-end. "
                "Use only --group-tag (and optional --fid-profile)."
            )
        version_start = 1
        version_end = 1
    else:
        if args.version_start is None or args.version_end is None:
            raise ValueError(
                "Legacy batch mode '{n}o{N}' requires --version-start and --version-end."
            )
        version_start = args.version_start
        version_end = args.version_end
        if version_start > version_end:
            raise ValueError("version-start must be <= version-end")

    print(f"Detected sample: {sample}")
    print(f"Detected optuna_mode: {mode}")
    print(f"Detected optuna_objective_index: {optuna_objective_index}")
    print(f"Detected optuna_n_trials: {optuna_n_trials}")
    print(f"Detected optuna_space_version: {optuna_space_version}")

    dag = make_dag(
        out_dir=Path(args.out_dir),
        group_tag=args.group_tag,
        mode=mode,
        version_start=version_start,
        version_end=version_end,
        optuna_n_trials=optuna_n_trials,
        fid_profile=args.fid_profile,
    )
    print(dag)


if __name__ == "__main__":
    main()
