#!/usr/bin/env python3
import argparse
from pathlib import Path

from utils.varsets import VARSET_COLUMNS, infer_varset_from_tag


def stage_group_for(version: int, version_token: str) -> str:
    if version_token in {"2v"}:
        return f"2v{((version - 1) % 10) + 1}"
    return f"{version_token}{version}"


def train_tag(group_tag: str, version: int, version_token: str) -> str:
    return f"{group_tag}_{version_token}{version}"


def validate_group_varset(group_tag: str) -> str:
    varset = infer_varset_from_tag(group_tag)
    if varset is None:
        raise ValueError(
            f"Unable to infer <varset> from group_tag '{group_tag}'. "
            f"Expected one of: {sorted(VARSET_COLUMNS.keys())}"
        )
    return varset


def make_dag(
    out_dir: Path,
    group_tag: str,
    version_start: int,
    version_end: int,
    version_token: str,
    optuna_n_trials: int,
    resume_flag: int,
    skip_version: int,
    draw_only: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_path = out_dir / f"wf_{group_tag}_{version_token}{version_start}_{version_token}{version_end}.dag"

    lines = []
    for version in range(version_start, version_end + 1):
        node = f"TR_{version_token}{version}"
        tag = train_tag(group_tag, version, version_token)
        stage_group = stage_group_for(version, version_token)
        job_tag = f"{tag}_staged"
        lines.append(f"JOB {node} submit_staged_single.sub")
        lines.append(
            f'VARS {node} train_tag="{tag}" stage_group="{stage_group}" '
            f'optuna_n_trials="{optuna_n_trials}" resume_flag="{resume_flag}" job_tag="{job_tag}"'
        )
        lines.append("")

    post_node = "POST_BATCH"
    post_job_tag = f"{group_tag}_{version_token}{version_start}_{version_token}{version_end}_batchcmp"
    lines.append(f"FINAL {post_node} submit_batch_compare_single.sub")
    lines.append(
        f'VARS {post_node} group_tag="{group_tag}" version_start="{version_start}" '
        f'version_end="{version_end}" skip_version="{skip_version}" draw_only="{draw_only}" '
        f'version_token="{version_token}" job_tag="{post_job_tag}"'
    )
    lines.append("")

    dag_path.write_text("\n".join(lines))
    return dag_path


def main():
    parser = argparse.ArgumentParser(description="Generate one DAG: parallel training + final batch apply/draw.")
    parser.add_argument("--group-tag", required=True)
    parser.add_argument("--version-start", type=int, required=True)
    parser.add_argument("--version-end", type=int, required=True)
    parser.add_argument("--version-token", default="v")
    parser.add_argument("--optuna-n-trials", type=int, default=200)
    parser.add_argument("--resume-flag", type=int, default=0)
    parser.add_argument("--skip-version", type=int, default=0)
    parser.add_argument("--draw-only", type=int, default=0)
    parser.add_argument("--out-dir", default="dags")
    args = parser.parse_args()

    if args.version_start > args.version_end:
        raise ValueError("version-start must be <= version-end")

    varset = validate_group_varset(args.group_tag)
    print(f"Detected varset: {varset}")
    print(f"Varset columns: {VARSET_COLUMNS[varset]}")

    dag = make_dag(
        out_dir=Path(args.out_dir),
        group_tag=args.group_tag,
        version_start=args.version_start,
        version_end=args.version_end,
        version_token=args.version_token,
        optuna_n_trials=args.optuna_n_trials,
        resume_flag=args.resume_flag,
        skip_version=args.skip_version,
        draw_only=args.draw_only,
    )
    print(dag)


if __name__ == "__main__":
    main()
