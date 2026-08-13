import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_reweight_profile,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_training_config,
    resolve_training_reweight_config,
    to_root_spec,
)
from utils.paths import resolve_model_config_path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "output/slides_evidence/a5_b1_b3_v1"
WEIGHTED_TAG = "X_pb24_v6_fid6_8v2_rwr6range2v1_xgb_v1"
UNWEIGHTED_TAG = "X_pb24_v6_fid6_8v2_xgb_v1"
H005 = REPO / "output/reweighting/diagnostics_pp24_psi2s_bootstrap_v1"
DOMAIN = REPO / "output/reweighting/X_pp24_psi2s_R5_rw_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare reproducible A5/B1--B3 evidence for the update slides."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_root_spec(spec):
    path, tree = spec.rsplit(":", 1)
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    return f"{path}:{tree}"


def resolve_tag(tag):
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    year = infer_dataset_year(tag, sample)
    selection_profile = infer_selection_profile(tag, sample)
    fid_profile = infer_fid_profile(tag, sample)
    reweight_profile = infer_reweight_profile(tag)
    training = resolve_training_config(sample, channel, year, selection_profile)
    reweight = resolve_training_reweight_config(
        sample, channel, year, reweight_profile, selection_profile, fid_profile
    )
    signal = reweight["signal"] if reweight["signal"] is not None else training["signal"]
    return {
        "sample": sample,
        "channel": channel,
        "year": year,
        "selection_profile": selection_profile,
        "fid_profile": fid_profile,
        "reweight_profile": reweight_profile,
        "signal": absolute_root_spec(to_root_spec(signal)),
        "background": absolute_root_spec(to_root_spec(training["background"])),
        "signal_selection": training["signal_selection"],
        "background_selection": training["background_selection"],
        "weight_branch": reweight["weight_branch"],
    }


def collect_shap(tag, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_tag(tag)
    model_config_path = REPO / resolve_model_config_path(tag)
    model_config = read_json(model_config_path)
    features = model_config["input_columns"]
    source_dir = REPO / "output/shap" / tag
    source_pdf = source_dir / "shap_cumulative.pdf"
    source_ranking = source_dir / "shap_importance_fraction.json"
    source_rows = read_json(source_ranking)
    rows = [
        {
            "rank": row["rank"],
            "feature": row["feature"],
            "mean_abs_shap": row["mean_abs_shap"],
            "fraction": row["fraction"],
            "cumulative_fraction": row["cumulative_percent"] / 100.0,
        }
        for row in source_rows
    ]
    shutil.copy2(source_pdf, output_dir / "shap_cumulative.pdf")

    write_json(output_dir / "shap_ranking.json", rows)
    with (output_dir / "shap_ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    retained_95 = []
    for row in rows:
        retained_95.append(row["feature"])
        if row["cumulative_fraction"] >= 0.95:
            break
    configuration = {
        "schema_version": 1,
        "train_tag": tag,
        "configuration_name": "PbPb R6range2-weighted 8v2" if resolved["weight_branch"] else "PbPb unweighted 8v2",
        "resolver": resolved,
        "model_config_path": str(model_config_path.relative_to(REPO)),
        "model_config_sha256": sha256(model_config_path),
        "features": features,
        "training_split": {
            "actual_policy": "candidate-level sklearn train_test_split",
            "train_fraction": 0.75,
            "test_fraction": 0.25,
            "random_state": 42,
            "note": "model training code and newer model_config define 75/25; legacy run_metadata fields 0.8/0.1/0.1 do not match the executed direct-training implementation",
        },
        "shap": {
            "explainer": "shap.TreeExplainer",
            "definition": "per-feature weighted mean absolute SHAP value over the sampled combined signal-MC and DATA-sideband candidates",
            "sampling": "uniform candidate sample after selections; weights enter only the mean-|SHAP| aggregation",
            "max_events": 20000,
            "random_state": 42,
            "sample_counts": "not persisted by the original SHAP job",
            "signal_weight_branch": resolved["weight_branch"],
            "background_weight": "unit",
            "cumulative_threshold": 0.95,
            "retained_at_95pct": retained_95,
            "source_pdf": str(source_pdf.relative_to(REPO)),
            "source_ranking": str(source_ranking.relative_to(REPO)),
            "source_pdf_sha256": sha256(source_pdf),
            "source_ranking_sha256": sha256(source_ranking),
        },
        "artifacts": {
            "pdf": "shap_cumulative.pdf",
            "ranking_json": "shap_ranking.json",
            "ranking_csv": "shap_ranking.csv",
        },
    }
    write_json(output_dir / "configuration_manifest.json", configuration)
    return configuration, rows


def prepare_b1(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "weight_distributions_full_range.pdf",
        "weight_tail_summary.json",
        "weight_tail_summary.csv",
        "bootstrap_weight_tail_summary.json",
        "bootstrap_weight_tail_summary.csv",
        "cdf_distance_bootstrap_summary.pdf",
        "bootstrap_summary.csv",
    ]
    for name in names:
        shutil.copy2(H005 / name, output_dir / name)
    return names


def prepare_b2(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    definition = {
        "schema_version": 1,
        "metric": "maximum signed-weighted empirical-CDF distance",
        "formula": "D=max_x |F_MC(x)-F_target_signed(x)|; F_w(x)=sum_i w_i 1[x_i<=x]/sum_i w_i",
        "implementation": "workflows/reweighting/core.py::weighted_cdf_distance",
        "normalization_requirement": "the total weight of each sample must be positive",
        "signed_target_behavior": "negative sWeights are retained; cumulative steps may decrease and the resulting signed empirical CDF need not be monotonic or remain inside [0,1]",
        "eight_variable_summary": "unweighted arithmetic mean of the eight per-variable distances; descriptive summary only",
        "standard_ks_statistic": False,
        "standard_ks_pvalue_available": False,
        "h005_bootstrap": {
            "replicas": 500,
            "fixed_split_seed": 314159,
            "base_replica_seed": 260729,
            "train_fraction": 0.7,
            "paired_profiles": True,
            "intervals": "empirical percentile intervals: central 68%=[q16,q84], central 95%=[q2.5,q97.5]",
            "covered": [
                "finite-event resampling fluctuations in DATA and MC train/holdout partitions",
                "reweighter retraining fluctuations",
            ],
            "not_covered": [
                "mass-fit model",
                "mass-fit range",
                "sWeight construction/model systematics",
                "other fit-side systematics",
            ],
        },
        "x_splot_transfer_status": "point estimate only; no uncertainty evaluation, so before/after differences are not statistically significant claims",
    }
    write_json(output_dir / "metric_definition.json", definition)
    shutil.copy2(H005 / "bootstrap_summary.csv", output_dir / "bootstrap_summary.csv")
    shutil.copy2(H005 / "manifest.json", output_dir / "h005_manifest.json")
    return definition


def prepare_b3(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_json(DOMAIN / "domain_classifier_holdout_stability.json")
    seeds = np.asarray(source["random_states"])
    before = np.asarray(source["before_signed_auc"])
    after = np.asarray(source["after_signed_auc"])
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.axhline(0.5, color="0.35", linestyle="--", label="Chance-level discrimination")
    axis.plot(seeds, before, "o-", color="#0072B2", label="MC before vs signed-sWeighted DATA")
    axis.plot(seeds, after, "o-", color="#D55E00", label="MC after R5 vs signed-sWeighted DATA")
    axis.set_xlabel("Three-way split random seed")
    axis.set_ylabel("Signed holdout AUC (target=DATA, source=MC)")
    axis.set_title("Holdout domain-classifier AUC: R5 correction, R8 validator")
    axis.set_ylim(0.45, 0.76)
    axis.legend(frameon=False, loc="best")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(output_dir / "holdout_domain_classifier_auc.pdf")
    plt.close(figure)

    seed_details = []
    for seed in seeds:
        seed_details.append(read_json(DOMAIN / f"domain_classifier_holdout_seed{int(seed)}.json"))
    metadata = {
        "schema_version": 1,
        "classification": {
            "positive_target": "ppRef psi(2S) DATA signal target with signed signal_sWeight",
            "negative_source": "ppRef psi(2S) MC, unit weight before or R5 weight after correction",
            "validator_variables": seed_details[0]["domain_classifier"]["variables"],
            "reweighter_variables": seed_details[0]["reweighter"]["variables"],
            "split": seed_details[0]["split"],
            "independence": "reweighter train, domain-classifier train, and domain test are disjoint within each split",
            "seed_relation": "domain-classifier random_state is split random_state + 10",
        },
        "metric": "signed weighted rank AUC; target label=1, MC label=0",
        "interpretation": "AUC=0.5 means chance-level DATA-vs-MC discrimination in this validation test; it does not mean random reweighting and does not alone establish complete closure",
        "variation_scope": "five split seeds only; no larger seed ensemble or systematic variation",
        "source": source,
        "per_seed_source_json": [str((DOMAIN / f"domain_classifier_holdout_seed{int(seed)}.json").relative_to(REPO)) for seed in seeds],
        "artifact": "holdout_domain_classifier_auc.pdf",
    }
    write_json(output_dir / "domain_auc_metadata.json", metadata)
    return metadata


def git_value(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def main():
    args = parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    output.mkdir(parents=True, exist_ok=True)

    weighted_config, weighted_rows = collect_shap(WEIGHTED_TAG, output / "a5/weighted")
    unweighted_config, unweighted_rows = collect_shap(UNWEIGHTED_TAG, output / "a5/unweighted")
    b1_files = prepare_b1(output / "b1")
    b2 = prepare_b2(output / "b2")
    b3 = prepare_b3(output / "b3")

    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[str(path.relative_to(output))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "study": "A5_B1_B2_B3_slides_evidence",
        "git": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "worktree_dirty": bool(git_value("status", "--short")),
        },
        "generation_command": ".venv/bin/python -m workflows.prepare_slides_reweighting_evidence",
        "old_slides_provenance": {
            "repository": "../pre-updates",
            "commit": "a13a9d7",
            "pdf": "../pre-updates/pbpb_updates/X_update_260729.pdf",
            "p8_source_in_tex": "../XGBoost/output/reweighting/comparison_pp24_psi2s/weight_distributions_R3_R4_R5_R8.pdf",
            "p10_source_in_tex": "../XGBoost/output/reweighting/comparison_pp24_psi2s/domain_auc_stability.pdf",
            "p10_issue": "embedded plot title says R8 domain-classifier stability although the correction is R5 and the validator variable set is R8",
        },
        "a5": {
            "weighted": weighted_config,
            "unweighted": unweighted_config,
            "weighted_ranking": weighted_rows,
            "unweighted_ranking": unweighted_rows,
        },
        "b1": {
            "source_manifest": str((H005 / "manifest.json").relative_to(REPO)),
            "copied_artifacts": b1_files,
        },
        "b2": b2,
        "b3": b3,
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Slides evidence: {output}")


if __name__ == "__main__":
    main()
