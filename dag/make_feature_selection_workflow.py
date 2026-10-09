"""Build one DAG for an appendable feature-selection experiment."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
from configs.feature_selection import resolve_feature_selection_experiment
from configs.samples import (infer_sample_from_tag, infer_channel_from_tag,
    infer_dataset_year, infer_selection_profile, resolve_training_config, default_apply_extra_mc)
from utils.paths import model_path, selected_dir, shap_importance_path, shap_importance_fraction_path, shap_summary_path, shap_bar_path, shap_cumulative_path, punzi_dir


def make_dag(experiment, out_dir):
    cfg = resolve_feature_selection_experiment(experiment)
    lines, parents = [], []
    items = [{"train_tag": cfg["baseline"]}] + cfg["candidates"]
    for index, item in enumerate(items):
        tag = item["train_tag"]
        sample = infer_sample_from_tag(tag)
        resolve_training_config(sample, infer_channel_from_tag(tag),
            infer_dataset_year(tag, sample), infer_selection_profile(tag, sample))
        # Reuse completed steps; an importance JSON alone is not a finished SHAP job.
        draw_complete = (Path(selected_dir(tag), "cut_scan", "thresholds.json").is_file()
                         and (not tag.startswith("X_pb23_") or Path(punzi_dir(tag), "scan.json").is_file()))
        shap_complete = all(Path(fn(tag)).is_file() for fn in (
            shap_importance_path, shap_importance_fraction_path, shap_summary_path,
            shap_bar_path, shap_cumulative_path))
        extra_mc = default_apply_extra_mc(tag)
        extra_needed = (extra_mc != "0" and not
                        Path(selected_dir(tag), "MC_psi2s_with_score.root").is_file())
        if draw_complete and shap_complete and Path(model_path(tag)).is_file() and not extra_needed:
            continue
        existing = Path(model_path(tag)).exists()
        if existing and not Path(selected_dir(tag), "batch_apply_summary.json").is_file():
            raise FileExistsError(f"Partial candidate exists: {tag}; resume its original DAG")
        steps = (([] if draw_complete else ["DRAW"]) + ([] if shap_complete else ["SHAP"])) if existing else ["TRAIN", "APPLY", "DRAW", "SHAP"]
        for step in steps:
            node = f"{step}_{index}"
            lines += [f"JOB {node} submit_{step.lower()}_job.sub",
                f'VARS {node} train_tag="{tag}" use_precut="0" apply_extra_mc="0" job_tag="{tag}_{step.lower()}"']
        if not existing:
            lines += [f"PARENT TRAIN_{index} CHILD APPLY_{index}",
                      f"PARENT APPLY_{index} CHILD DRAW_{index}"]
        if "DRAW" in steps and "SHAP" in steps:
            lines += [f"PARENT DRAW_{index} CHILD SHAP_{index}"]
        if steps:
            parents.append(f"{steps[-1]}_{index}")
        if extra_needed:
            extra_node = f"PSI_{index}"
            lines += [f"JOB {extra_node} submit_apply_job.sub",
                f'VARS {extra_node} train_tag="{tag}" use_precut="0" '
                f'apply_extra_mc="{extra_mc}" job_tag="{tag}_psi2s"']
            if not existing:
                lines.append(f"PARENT TRAIN_{index} CHILD {extra_node}")
            if "DRAW" in steps:
                lines.append(f"PARENT {extra_node} CHILD DRAW_{index}")
            parents.append(extra_node)
    lines += ["JOB RANK submit_feature_selection_rank.sub",
              f'VARS RANK experiment="{experiment}" job_tag="{experiment}_rank"']
    if parents:
        lines.append("PARENT " + " ".join(parents) + " CHILD RANK")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out / f"wf_{experiment}_{stamp}.dag"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--out-dir", default="dag/generated")
    args = parser.parse_args()
    print(make_dag(args.experiment, args.out_dir))
