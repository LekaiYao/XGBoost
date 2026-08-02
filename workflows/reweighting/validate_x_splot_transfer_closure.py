import argparse
import csv
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

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


STATUS = "preliminary - uncertainty not evaluated"
VALIDATION_VARIABLES = [
    "Bchi2Prob",
    "Btrk1dR",
    "BtrkPtimb",
    "Btrk1Pt",
    "Btrk2Pt",
    "BtktkvProb",
    "Bcos_dtheta",
    "Btktkpt",
]
DEFAULT_SELECTION = "Bpt > 7.5 and Bpt < 50 and abs(By) < 2.4 and BQvalue < 0.15"
DEFAULT_DATA_ROOT = (
    "/eos/home-l/leyao/pbpb_work/X_analysis/Analysis_CODES/plotER/Validation/"
    "results/ppRef_X_r5_splot/"
    "SignalWeight_TTree_ppRef_X_r5_fiducial_splot_ntmix_X3872.root"
)
DEFAULT_MC_ROOT = (
    "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/"
    "flat_ntmix_ppRef_MC_X3872.root"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Point-estimate ppRef X signed-sWeight closure for transferred R5 weights."
    )
    parser.add_argument("reweight_tag", nargs="?", default="X_pp24_psi2s_R5_rw_v1")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-tree", default="ntmix_X3872_sWeight")
    parser.add_argument("--data-weight", default="signal_sWeight")
    parser.add_argument("--mc-root", default=DEFAULT_MC_ROOT)
    parser.add_argument("--mc-tree", default="ntmix_X3872")
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--output-dir")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Add the preliminary status to existing CSV/binning artifacts without recomputing.",
    )
    return parser.parse_args()


def quantile_bins(values, count=20):
    values = np.asarray(values, dtype=float)
    if count < 2:
        raise ValueError("At least two bins are required")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Binning values must be nonempty and finite")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, count + 1)))
    if len(edges) < 3:
        raise ValueError("Variable has fewer than two nonempty quantile bins")
    return edges


def normalized_histogram(values, weights, bins):
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Normalized signed histogram requires positive total weight")
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, weights=np.square(weights))
    return counts / total, np.sqrt(sumw2) / abs(total)


def ratio_to_signed_target(numerator, target):
    numerator = np.asarray(numerator, dtype=float)
    target = np.asarray(target, dtype=float)
    threshold = max(1e-12, 1e-6 * float(np.max(np.abs(target))))
    valid = np.abs(target) > threshold
    ratio = np.full(target.shape, np.nan, dtype=float)
    ratio[valid] = numerator[valid] / target[valid]
    return ratio, valid, threshold


def extended_weight_summary(weights):
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(weights)
    if not finite.all():
        raise ValueError("Weights contain non-finite values")
    summary = weight_summary(weights)
    summary["nonfinite_entries"] = int(np.count_nonzero(~finite))
    return summary


def plot_variable(variable, mc, data, mc_before, mc_after, data_weight, bins, metrics, output):
    centers = 0.5 * (bins[:-1] + bins[1:])
    before, before_error = normalized_histogram(mc[variable], mc_before, bins)
    after, after_error = normalized_histogram(mc[variable], mc_after, bins)
    target, target_error = normalized_histogram(data[variable], data_weight, bins)
    before_ratio, before_valid, threshold = ratio_to_signed_target(before, target)
    after_ratio, after_valid, _ = ratio_to_signed_target(after, target)
    valid = before_valid & after_valid

    figure, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(7.2, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    axis.errorbar(
        centers, target, yerr=target_error, fmt="o", color="black",
        markersize=3.5, label="X DATA signed sWeight",
    )
    axis.stairs(before, bins, color="#4477AA", label="X MC before")
    axis.stairs(after, bins, color="#CC6677", label="X MC after R5")
    axis.set_ylabel("Normalized weighted entries")
    axis.set_title(
        f"ppRef X transfer closure: {variable}\n"
        f"CDF distance {metrics['before']:.3f} -> {metrics['after']:.3f}"
    )
    axis.legend(frameon=False)
    axis.text(0.02, 0.96, STATUS, transform=axis.transAxes, fontsize=9, va="top")

    ratio_axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ratio_axis.plot(centers[valid], before_ratio[valid], "o", color="#4477AA", markersize=3)
    ratio_axis.plot(centers[valid], after_ratio[valid], "o", color="#CC6677", markersize=3)
    ratio_axis.set_ylabel("MC / DATA")
    ratio_axis.set_xlabel(variable)
    finite_ratio = np.concatenate([before_ratio[valid], after_ratio[valid]])
    if finite_ratio.size:
        low, high = np.quantile(finite_ratio, [0.02, 0.98])
        span = max(high - low, 0.5)
        ratio_axis.set_ylim(low - 0.2 * span, high + 0.2 * span)
    ratio_axis.text(
        0.02, 0.04, f"undefined bins: {len(valid) - np.count_nonzero(valid)}",
        transform=ratio_axis.transAxes, fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return {
        "edges": bins.tolist(),
        "target_zero_threshold": threshold,
        "undefined_ratio_bins": int(len(valid) - np.count_nonzero(valid)),
        "data_signed": target.tolist(),
        "data_signed_error": target_error.tolist(),
        "mc_before": before.tolist(),
        "mc_before_error": before_error.tolist(),
        "mc_after": after.tolist(),
        "mc_after_error": after_error.tolist(),
    }


def package_versions():
    packages = {}
    for name in ("numpy", "pandas", "uproot", "hep_ml", "scikit-learn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def refresh_metadata(output_dir):
    csv_path = output_dir / "transfer_closure_summary.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = list(rows[0]) if rows else []
    if "status" not in fieldnames:
        fieldnames.insert(0, "status")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["status"] = STATUS
            writer.writerow(row)

    binning_path = output_dir / "binning_and_histograms.json"
    binning = json.loads(binning_path.read_text())
    if "variables" not in binning:
        binning = {"status": STATUS, "variables": binning}
    else:
        binning["status"] = STATUS
    write_json(binning_path, binning)


def main():
    args = parse_args()
    reweight_dir = Path("output/reweighting") / args.reweight_tag
    output_dir = Path(
        args.output_dir
        or reweight_dir / "transfer_closure_x_splot_v1"
    )
    if args.metadata_only:
        refresh_metadata(output_dir)
        print(f"Transfer closure metadata refreshed: {output_dir}")
        return
    source_manifest_path = reweight_dir / "reweighting_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["variable_set"] != "R5":
        raise ValueError("This closure request requires the nominal R5 reweighter")
    model = joblib.load(reweight_dir / source_manifest["artifacts"]["model"])
    r5_variables = source_manifest["variables"]
    required = list(dict.fromkeys([
        *VALIDATION_VARIABLES, *r5_variables, "Bpt", "By", "BQvalue"
    ]))

    data = select_frame(load_tree_frame(args.data_root, args.data_tree), args.selection, "DATA selection")
    mc = select_frame(load_tree_frame(args.mc_root, args.mc_tree), args.selection, "MC selection")
    validate_columns(data, [*required, args.data_weight], "X DATA")
    validate_columns(mc, required, "X MC")
    data_weight = resolve_weights(data, args.data_weight, "X DATA")
    mc_before = np.ones(len(mc), dtype=float)
    mc_after = predict_reweight(model, mc, r5_variables, mc_before)
    if not np.isfinite(mc_after).all() or np.any(mc_after <= 0.0):
        raise ValueError("Transferred R5 weights must be finite and positive")

    plot_dir = output_dir / "variables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_variable = {}
    binning = {}
    for variable in VALIDATION_VARIABLES:
        bins = quantile_bins(mc[variable], args.bins)
        before = weighted_cdf_distance(mc[variable], data[variable], mc_before, data_weight)
        after = weighted_cdf_distance(mc[variable], data[variable], mc_after, data_weight)
        per_variable[variable] = {
            "before": before,
            "after": after,
            "change_before_minus_after": before - after,
        }
        binning[variable] = plot_variable(
            variable, mc, data, mc_before, mc_after, data_weight, bins,
            per_variable[variable], plot_dir / f"{variable}_transfer_closure.pdf",
        )

    mean_before = float(np.mean([item["before"] for item in per_variable.values()]))
    mean_after = float(np.mean([item["after"] for item in per_variable.values()]))
    summary = {
        "schema_version": 1,
        "status": STATUS,
        "metric": "maximum signed-weighted empirical-CDF distance",
        "metric_is_standard_ks_pvalue": False,
        "per_variable": per_variable,
        "arithmetic_mean": {
            "before": mean_before,
            "after": mean_after,
            "change_before_minus_after": mean_before - mean_after,
        },
        "weights": {
            "data_signed_sweight": extended_weight_summary(data_weight),
            "mc_before": extended_weight_summary(mc_before),
            "mc_after_r5": extended_weight_summary(mc_after),
        },
        "entries": {"data": len(data), "mc": len(mc)},
    }
    write_json(output_dir / "transfer_closure_summary.json", summary)
    with (output_dir / "transfer_closure_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "status", "variable", "cdf_distance_before", "cdf_distance_after",
            "change_before_minus_after", "bin_count", "undefined_ratio_bins",
        ])
        for variable in VALIDATION_VARIABLES:
            item = per_variable[variable]
            writer.writerow([
                STATUS, variable, item["before"], item["after"],
                item["change_before_minus_after"],
                len(binning[variable]["edges"]) - 1,
                binning[variable]["undefined_ratio_bins"],
            ])
        writer.writerow([STATUS, "arithmetic_mean", mean_before, mean_after, mean_before - mean_after, "", ""])
    write_json(
        output_dir / "binning_and_histograms.json",
        {"status": STATUS, "variables": binning},
    )

    manifest = {
        "schema_version": 1,
        "study": "ppRef_X_R5_transfer_closure_point_estimate",
        "status": STATUS,
        "reweight_tag": args.reweight_tag,
        "reweight_model": str((reweight_dir / source_manifest["artifacts"]["model"]).resolve()),
        "source_reweight_manifest": str(source_manifest_path.resolve()),
        "code_commit": args.code_commit,
        "inputs": {
            "data": {"path": str(Path(args.data_root).resolve()), "tree": args.data_tree, "weight": args.data_weight},
            "mc": {"path": str(Path(args.mc_root).resolve()), "tree": args.mc_tree, "weight": "unit before; transferred R5 after"},
        },
        "selection": args.selection,
        "reweight_variables": r5_variables,
        "validation_variables": VALIDATION_VARIABLES,
        "binning": {
            "method": "equal-occupancy quantiles of selected unweighted X MC",
            "requested_bins": args.bins,
            "full_selected_MC_range_included": True,
            "edges_artifact": "binning_and_histograms.json",
        },
        "weight_definitions": {
            "data": "signed signal_sWeight, unchanged",
            "mc_before": "unit weight",
            "mc_after": "R5 model prediction multiplied by unit original weight",
        },
        "software": package_versions(),
        "artifacts": {
            "summary_json": "transfer_closure_summary.json",
            "summary_csv": "transfer_closure_summary.csv",
            "binning_histograms_json": "binning_and_histograms.json",
            "variable_plots": "variables/*.pdf",
            "interpretation": "interpretation.md",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    interpretation = [
        "# ppRef X R5 transfer-closure point estimate",
        "",
        f"**{STATUS}.**",
        "",
        "The maximum signed-weighted empirical-CDF distance is descriptive and is not a KS p-value.",
        "No bootstrap or mass-fit/sWeight systematic variation is included, so before/after changes",
        "must not be interpreted as statistically significant improvements.",
        "",
        f"The arithmetic mean changes from {mean_before:.6g} before to {mean_after:.6g} after R5.",
        "Per-variable changes and signed-weight stability are recorded in the JSON/CSV artifacts.",
        "The low signed-DATA effective sample size remains the principal interpretation caveat.",
    ]
    (output_dir / "interpretation.md").write_text("\n".join(interpretation) + "\n")
    print(f"Transfer closure: {output_dir}")


if __name__ == "__main__":
    main()
