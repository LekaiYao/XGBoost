#!/usr/bin/env python3
import argparse
from pathlib import Path

from workflows.reweighting.run_configured_job import (
    configured_apply_jobs,
    resolve_training_job,
    validation_splot_spec,
)


def make_dag(
    out_dir: Path,
    reweight_tag: str,
    with_splot_validation: bool = True,
    with_mc_domain_validation: bool = True,
):
    training_spec = resolve_training_job(reweight_tag)
    apply_jobs = configured_apply_jobs(reweight_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_path = out_dir / f"wf_reweight_{reweight_tag}.dag"
    lines = [
        "JOB TRAIN submit_reweight_job.sub",
        f'VARS TRAIN reweight_job="{reweight_tag}" stage="train" job_tag="{reweight_tag}_train"',
        "",
    ]
    for apply_job, spec in apply_jobs:
        year = spec["apply_dataset_year"]
        node = f"APPLY_{year}"
        lines.extend([
            f"JOB {node} submit_reweight_job.sub",
            f'VARS {node} reweight_job="{apply_job}" stage="apply" job_tag="{reweight_tag}_apply_{year}"',
            "",
            f"PARENT TRAIN CHILD {node}",
            "",
        ])
        if with_mc_domain_validation:
            validation_node = f"VALIDATE_MC_{year}"
            lines.extend([
                f"JOB {validation_node} submit_reweight_validation.sub",
                f'VARS {validation_node} reweight_tag="{reweight_tag}" validation_kind="ppref-pbpb" '
                f'apply_job="{apply_job}" job_tag="{reweight_tag}_validate_mc_{year}"',
                "",
                f"PARENT TRAIN CHILD {validation_node}",
                "",
            ])
    if with_splot_validation:
        lines.extend([
            "JOB VALIDATE_SPLOT submit_reweight_validation.sub",
            f'VARS VALIDATE_SPLOT reweight_tag="{reweight_tag}" validation_kind="splot" '
            f'apply_job="none" job_tag="{reweight_tag}_validate_splot"',
            "",
            "PARENT TRAIN CHILD VALIDATE_SPLOT",
            "",
        ])
    dag_path.write_text("\n".join(lines))
    splot = validation_splot_spec(training_spec)
    return dag_path, Path(splot.get("path") or "").is_file(), splot.get("path")


def main():
    parser = argparse.ArgumentParser(description="Generate pp reweight train/apply/validation DAG.")
    parser.add_argument("--reweight-tag", required=True)
    parser.add_argument("--with-splot-validation", type=int, choices=(0, 1), default=1)
    parser.add_argument("--with-mc-domain-validation", type=int, choices=(0, 1), default=1)
    parser.add_argument("--out-dir", default="dag/generated")
    args = parser.parse_args()
    dag, splot_exists, splot_path = make_dag(
        Path(args.out_dir), args.reweight_tag,
        bool(args.with_splot_validation), bool(args.with_mc_domain_validation),
    )
    if args.with_splot_validation:
        state = "available" if splot_exists else "missing; validation node will auto-skip"
        print(f"sPlot preflight: {state}: {splot_path}")
    print(dag)


if __name__ == "__main__":
    main()
