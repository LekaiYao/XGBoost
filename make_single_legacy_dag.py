#!/usr/bin/env python3
import argparse
from pathlib import Path


def make_dag(out_dir: Path, train_tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_path = out_dir / f"wf_single_{train_tag}.dag"

    lines = [
        "JOB TRAIN submit_single_legacy_step.sub",
        f'VARS TRAIN step="train" train_tag="{train_tag}" job_tag="{train_tag}_single_train"',
        "",
        "JOB APPLY submit_single_legacy_step.sub",
        f'VARS APPLY step="apply" train_tag="{train_tag}" job_tag="{train_tag}_single_apply"',
        "",
        "JOB DRAW submit_single_legacy_step.sub",
        f'VARS DRAW step="draw" train_tag="{train_tag}" job_tag="{train_tag}_single_draw"',
        "",
        "PARENT TRAIN CHILD APPLY",
        "PARENT APPLY CHILD DRAW",
        "",
    ]

    dag_path.write_text("\n".join(lines))
    return dag_path


def main():
    parser = argparse.ArgumentParser(description="Generate DAG for single-model legacy train->apply->draw.")
    parser.add_argument("--train-tag", required=True)
    parser.add_argument("--out-dir", default="dags")
    args = parser.parse_args()

    dag = make_dag(Path(args.out_dir), args.train_tag)
    print(dag)


if __name__ == "__main__":
    main()
