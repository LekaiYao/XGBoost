#!/usr/bin/env python3
import argparse
from pathlib import Path

from utils.tagging import split_channel_tag


def make_dag(
    out_dir: Path,
    train_tag: str,
    with_shap: bool,
    use_precut: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_path = out_dir / f"wf_single_{train_tag}.dag"
    precut_flag = int(use_precut)

    lines = [
        "JOB TRAIN submit_train_job.sub",
        f'VARS TRAIN train_tag="{train_tag}" use_precut="{precut_flag}" job_tag="{train_tag}_train"',
        "",
        "JOB APPLY submit_apply_job.sub",
        f'VARS APPLY train_tag="{train_tag}" use_precut="{precut_flag}" apply_extra_mc="0" job_tag="{train_tag}_apply"',
        "",
        "JOB DRAW submit_draw_job.sub",
        f'VARS DRAW train_tag="{train_tag}" job_tag="{train_tag}_draw"',
        "",
        "PARENT TRAIN CHILD APPLY",
        "PARENT APPLY CHILD DRAW",
        "",
    ]
    if with_shap:
        lines.extend(
            [
                "JOB SHAP submit_shap_job.sub",
                f'VARS SHAP train_tag="{train_tag}" use_precut="{precut_flag}" job_tag="{train_tag}_shap"',
                "",
                "PARENT DRAW CHILD SHAP",
                "",
            ]
        )

    dag_path.write_text("\n".join(lines))
    return dag_path


def main():
    parser = argparse.ArgumentParser(description="Generate DAG for single-tag train->apply->draw (optional shap).")
    parser.add_argument("--train-tag", required=True)
    parser.add_argument("--with-shap", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use-precut", type=int, choices=[0, 1], default=0)
    parser.add_argument("--out-dir", default="dag/generated")
    args = parser.parse_args()
    split_channel_tag(args.train_tag)

    dag = make_dag(Path(args.out_dir), args.train_tag, bool(args.with_shap), bool(args.use_precut))
    print(dag)


if __name__ == "__main__":
    main()
