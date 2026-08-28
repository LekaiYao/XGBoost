import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from configs.samples import resolve_training_config
from utils.paths import reweighting_dir
from utils.selection import selection_columns
from workflows.reweighting.core import (
    load_tree_frame,
    predict_reweight,
    resolve_weights,
    select_frame,
    validate_columns,
    weight_summary,
    weighted_cdf_distance,
    write_json,
)
from workflows.reweighting.run_configured_job import (
    JOBS,
    resolve_training_job,
    validation_splot_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_COUNTS = (10,)


def normalized_histogram(values, weights, bins):
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, weights=np.square(weights))
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Normalized histogram requires a positive total weight")
    return counts / total, np.sqrt(sumw2) / abs(total)


def common_bins(left, right, count):
    values = np.concatenate([np.asarray(left, dtype=float), np.asarray(right, dtype=float)])
    low, high = float(np.min(values)), float(np.max(values))
    if not high > low:
        raise ValueError("Cannot build bins for a constant variable")
    return np.linspace(low, high, count + 1)


def plot_comparison(variable, left, right, left_weight, right_weight, bins, labels, output):
    left_hist, left_error = normalized_histogram(left[variable], left_weight, bins)
    right_hist, right_error = normalized_histogram(right[variable], right_weight, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    figure, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(7.2, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.05},
    )
    axis.stairs(left_hist, bins, color="#4477AA", label=labels[0])
    axis.stairs(right_hist, bins, color="#CC6677", label=labels[1])
    axis.set_ylabel("Normalized entries")
    axis.set_title(variable)
    axis.legend(frameon=False)
    threshold = max(1e-12, 1e-6 * float(np.max(np.abs(right_hist))))
    valid = np.abs(right_hist) > threshold
    ratio = np.full_like(left_hist, np.nan)
    ratio_error = np.full_like(left_hist, np.nan)
    ratio[valid] = left_hist[valid] / right_hist[valid]
    ratio_error[valid] = np.sqrt(
        np.square(left_error[valid] / right_hist[valid])
        + np.square(
            left_hist[valid] * right_error[valid] / np.square(right_hist[valid])
        )
    )
    ratio_axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ratio_axis.errorbar(
        centers[valid], ratio[valid], yerr=ratio_error[valid], fmt="o",
        color="#4477AA", markersize=3, capsize=2, linewidth=1,
    )
    ratio_axis.set_ylabel("left/right")
    ratio_axis.set_xlabel(variable)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return {
        "edges": bins.tolist(),
        "left": left_hist.tolist(),
        "left_error": left_error.tolist(),
        "right": right_hist.tolist(),
        "right_error": right_error.tolist(),
        "ratio": [float(value) if np.isfinite(value) else None for value in ratio],
        "ratio_error": [
            float(value) if np.isfinite(value) else None for value in ratio_error
        ],
        "undefined_ratio_bins": int(np.count_nonzero(~valid)),
    }


def validate_splot(reweight_tag):
    spec = resolve_training_job(reweight_tag)
    output = REPO_ROOT / reweighting_dir(reweight_tag) / "validation" / "splot_vs_rw"
    splot = validation_splot_spec(spec)
    splot_path = Path(splot.get("path") or "")
    if not splot_path.is_file():
        payload = {
            "schema_version": 1,
            "validation": "splot_vs_rw",
            "status": "skipped_missing_splot",
            "missing_input": str(splot_path),
        }
        write_json(output / "manifest.json", payload)
        print(f"SKIP sPlot/RW validation: missing {splot_path}")
        return

    manifest_path = REPO_ROOT / reweighting_dir(reweight_tag) / "reweighting_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    variables = list(manifest["validation_variables"])
    selection = manifest["selection"]
    required = list(dict.fromkeys([*variables, *selection_columns(selection)]))
    original_spec = manifest["inputs"]["original"]
    print(f"Loading original MC: {original_spec['path']} [{original_spec['tree']}]", flush=True)
    original = select_frame(
        load_tree_frame(original_spec["path"], original_spec["tree"], required),
        selection,
        "sPlot/RW original selection",
    )
    print(f"Loaded selected original MC entries: {len(original)}", flush=True)
    print(f"Loading sPlot DATA: {splot_path} [{splot['tree']}]", flush=True)
    target = select_frame(
        load_tree_frame(
            splot_path, splot["tree"],
            list(dict.fromkeys([*required, splot["weight_branch"]])),
        ),
        selection,
        "sPlot/RW target selection",
    )
    print(f"Loaded selected sPlot DATA entries: {len(target)}", flush=True)
    validate_columns(original, required, "sPlot/RW original MC")
    validate_columns(target, [*required, splot["weight_branch"]], "sPlot DATA")
    original_weight = resolve_weights(original, original_spec.get("weight_branch"), "original MC")
    target_weight = resolve_weights(target, splot["weight_branch"], "sPlot DATA")
    model = joblib.load(REPO_ROOT / reweighting_dir(reweight_tag) / manifest["artifacts"]["model"])
    print("Applying trained reweighter and producing validation plots", flush=True)
    corrected = predict_reweight(model, original, manifest["variables"], original_weight)
    metrics, histograms = {}, {}
    for variable in variables:
        metrics[variable] = {
            "cdf_before": weighted_cdf_distance(
                original[variable], target[variable], original_weight, target_weight
            ),
            "cdf_after": weighted_cdf_distance(
                original[variable], target[variable], corrected, target_weight
            ),
        }
        histograms[variable] = {}
        for count in BIN_COUNTS:
            bins = common_bins(original[variable], target[variable], count)
            histograms[variable][str(count)] = plot_comparison(
                variable, original, target, corrected, target_weight, bins,
                ("RW MC", "signed-sWeight DATA"),
                output / f"{count}bin" / f"{variable}.pdf",
            )
    payload = {
        "schema_version": 1,
        "validation": "splot_vs_rw",
        "status": "complete_point_estimate",
        "reweight_tag": reweight_tag,
        "source_manifest": str(manifest_path),
        "selection": selection,
        "inputs": {"mc": original_spec, "splot": splot},
        "variables": variables,
        "bin_counts": list(BIN_COUNTS),
        "metric": "maximum signed-weighted empirical-CDF distance; descriptive, not a KS p-value",
        "metrics": metrics,
        "weights": {
            "mc_before": weight_summary(original_weight),
            "mc_after": weight_summary(corrected),
            "splot": weight_summary(target_weight),
        },
        "histograms": histograms,
    }
    write_json(output / "summary.json", payload)
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "validation": "splot_vs_rw",
        "status": "complete_point_estimate",
        "summary": "summary.json",
    })
    print(f"sPlot/RW validation: {output}")


def validate_ppref_pbpb(reweight_tag, apply_job):
    training_spec = resolve_training_job(reweight_tag)
    apply_spec = JOBS[apply_job]
    if apply_spec.get("source_reweight_tag", reweight_tag) != reweight_tag:
        raise ValueError(f"Apply job '{apply_job}' does not use '{reweight_tag}'")
    year = apply_spec["apply_dataset_year"]
    output = (
        REPO_ROOT / reweighting_dir(reweight_tag) / "validation"
        / "ppref_vs_pbpb_mc" / f"pb{year[-2:]}"
    )
    manifest_path = REPO_ROOT / reweighting_dir(reweight_tag) / "reweighting_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    variables = list(manifest["validation_variables"])
    pp_spec = resolve_training_config(
        training_spec["sample"], training_spec["channel"],
        training_spec["dataset_year"], training_spec["selection_profile"],
    )["signal"]
    pb_spec = resolve_training_config(
        apply_spec["apply_sample"], apply_spec["channel"], year,
        apply_spec["apply_selection_profile"],
    )["signal"]
    scenarios = {"reweight_support": manifest["selection"]}
    all_selection_columns = []
    for selection in scenarios.values():
        all_selection_columns.extend(selection_columns(selection))
    required = list(dict.fromkeys([*variables, *all_selection_columns]))
    print(f"Loading ppRef MC: {pp_spec['path']} [{pp_spec['tree']}]", flush=True)
    pp_all = load_tree_frame(pp_spec["path"], pp_spec["tree"], required)
    print(f"Loaded ppRef MC entries: {len(pp_all)}", flush=True)
    print(f"Loading PbPb {year} MC: {pb_spec['path']} [{pb_spec['tree']}]", flush=True)
    pb_all = load_tree_frame(pb_spec["path"], pb_spec["tree"], required)
    print(f"Loaded PbPb {year} MC entries: {len(pb_all)}", flush=True)
    validate_columns(pp_all, required, "ppRef MC")
    validate_columns(pb_all, required, f"PbPb {year} MC")
    summaries = {}
    for scenario, selection in scenarios.items():
        pp = select_frame(pp_all, selection, f"{scenario} ppRef selection")
        pb = select_frame(pb_all, selection, f"{scenario} PbPb selection")
        pp_weight = np.ones(len(pp), dtype=float)
        pb_weight = np.ones(len(pb), dtype=float)
        per_variable, histograms = {}, {}
        for variable in variables:
            per_variable[variable] = {
                "cdf_distance": weighted_cdf_distance(
                    pp[variable], pb[variable], pp_weight, pb_weight
                )
            }
            histograms[variable] = {}
            for count in BIN_COUNTS:
                bins = common_bins(pp[variable], pb[variable], count)
                histograms[variable][str(count)] = plot_comparison(
                    variable, pp, pb, pp_weight, pb_weight, bins,
                    ("ppRef MC", f"PbPb {year} MC"),
                    output / scenario / f"{count}bin" / f"{variable}.pdf",
                )
        summaries[scenario] = {
            "selection": selection,
            "entries": {"ppref": len(pp), "pbpb": len(pb)},
            "per_variable": per_variable,
            "histograms": histograms,
        }
    payload = {
        "schema_version": 1,
        "validation": "ppref_vs_pbpb_mc",
        "status": "complete_point_estimate",
        "reweight_tag": reweight_tag,
        "dataset_year": year,
        "source_manifest": str(manifest_path),
        "inputs": {"ppref": pp_spec, "pbpb": pb_spec},
        "variables": variables,
        "bin_counts": list(BIN_COUNTS),
        "metric": "maximum empirical-CDF distance; descriptive, not a KS p-value",
        "scenarios": summaries,
    }
    write_json(output / "summary.json", payload)
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "validation": "ppref_vs_pbpb_mc",
        "status": "complete_point_estimate",
        "summary": "summary.json",
    })
    print(f"ppRef/PbPb validation: {output}")


def main():
    parser = argparse.ArgumentParser(description="Run configured external reweighting validation.")
    parser.add_argument("reweight_tag")
    parser.add_argument("kind", choices=("splot", "ppref-pbpb"))
    parser.add_argument("--apply-job")
    args = parser.parse_args()
    if args.kind == "splot":
        validate_splot(args.reweight_tag)
    else:
        if not args.apply_job:
            parser.error("--apply-job is required for ppref-pbpb validation")
        validate_ppref_pbpb(args.reweight_tag, args.apply_job)


if __name__ == "__main__":
    main()
