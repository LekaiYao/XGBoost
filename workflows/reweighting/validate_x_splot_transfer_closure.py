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
AUXILIARY_VALIDATION_VARIABLES = ["Btrk2dR"]
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
    parser.add_argument(
        "--selection",
        help="Selection override; defaults to the source reweighter manifest selection",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--btrk2dr-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Optional Btrk2dR-only closure range; does not alter the eight-variable mean sample",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Add the preliminary status to existing CSV/binning artifacts without recomputing.",
    )
    return parser.parse_args()


def common_equal_width_bins(*samples, count=10):
    values = np.concatenate([np.asarray(sample, dtype=float) for sample in samples])
    if count < 2:
        raise ValueError("At least two bins are required")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Common binning values must be nonempty and finite")
    low = float(values.min())
    high = float(values.max())
    if not high > low:
        raise ValueError("Variable has no finite range for common binning")
    return np.linspace(low, high, count + 1)


def normalized_histogram(values, weights, bins):
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Normalized signed histogram requires positive total weight")
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, weights=np.square(weights))
    widths = np.diff(bins)
    return counts / (total * widths), np.sqrt(sumw2) / (abs(total) * widths)


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


def plot_variable(
    variable, mc, data, mc_before, mc_after, data_weight, bins, metrics, output,
    reweight_label,
):
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
    axis.stairs(after, bins, color="#CC6677", label=f"X MC after {reweight_label}")
    axis.set_ylabel("Normalized weighted density")
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
        "histogram_quantity": "normalized density per unit variable",
        "data_signed_density": target.tolist(),
        "data_signed_density_error": target_error.tolist(),
        "mc_before_density": before.tolist(),
        "mc_before_density_error": before_error.tolist(),
        "mc_after_density": after.tolist(),
        "mc_after_density_error": after_error.tolist(),
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
    selection = args.selection or source_manifest.get("selection", DEFAULT_SELECTION)
    reweight_label = source_manifest["variable_set"]
    if reweight_label not in {"R5", "R5v2", "R6", "R6v2"}:
        raise ValueError("This closure supports R5, R5v2, R6, or R6v2 reweighters")
    model = joblib.load(reweight_dir / source_manifest["artifacts"]["model"])
    reweight_variables = source_manifest["variables"]
    required = list(dict.fromkeys([
        *VALIDATION_VARIABLES, *AUXILIARY_VALIDATION_VARIABLES,
        *reweight_variables, "Bpt", "By", "BQvalue"
    ]))

    data = select_frame(
        load_tree_frame(
            args.data_root,
            args.data_tree,
            columns=list(dict.fromkeys([*required, args.data_weight])),
        ),
        selection,
        "DATA selection",
    )
    mc = select_frame(
        load_tree_frame(args.mc_root, args.mc_tree, columns=required),
        selection,
        "MC selection",
    )
    validate_columns(data, [*required, args.data_weight], "X DATA")
    validate_columns(mc, required, "X MC")
    data_weight = resolve_weights(data, args.data_weight, "X DATA")
    mc_before = np.ones(len(mc), dtype=float)
    mc_after = predict_reweight(model, mc, reweight_variables, mc_before)
    if not np.isfinite(mc_after).all() or np.any(mc_after <= 0.0):
        raise ValueError("Transferred reweighting weights must be finite and positive")

    plot_dir = output_dir / "variables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_variable = {}
    binning = {}
    for variable in [*VALIDATION_VARIABLES, *AUXILIARY_VALIDATION_VARIABLES]:
        variable_mc = mc
        variable_data = data
        variable_mc_before = mc_before
        variable_mc_after = mc_after
        variable_data_weight = data_weight
        evaluation_range = None
        if variable == "Btrk2dR" and args.btrk2dr_range is not None:
            low, high = args.btrk2dr_range
            if not high > low:
                raise ValueError("Btrk2dR closure range requires HIGH > LOW")
            mc_mask = mc[variable].between(low, high, inclusive="both").to_numpy()
            data_mask = data[variable].between(low, high, inclusive="both").to_numpy()
            variable_mc = mc.loc[mc_mask]
            variable_data = data.loc[data_mask]
            variable_mc_before = mc_before[mc_mask]
            variable_mc_after = mc_after[mc_mask]
            variable_data_weight = data_weight[data_mask]
            evaluation_range = [low, high]
        bins = common_equal_width_bins(
            variable_mc[variable], variable_data[variable], count=args.bins
        )
        before = weighted_cdf_distance(
            variable_mc[variable], variable_data[variable],
            variable_mc_before, variable_data_weight,
        )
        after = weighted_cdf_distance(
            variable_mc[variable], variable_data[variable],
            variable_mc_after, variable_data_weight,
        )
        per_variable[variable] = {
            "before": before,
            "after": after,
            "change_before_minus_after": before - after,
            "evaluation_range": evaluation_range,
            "entries": {"data": len(variable_data), "mc": len(variable_mc)},
        }
        binning[variable] = plot_variable(
            variable, variable_mc, variable_data,
            variable_mc_before, variable_mc_after, variable_data_weight, bins,
            per_variable[variable], plot_dir / f"{variable}_transfer_closure.pdf",
            reweight_label,
        )

    mean_before = float(np.mean([per_variable[name]["before"] for name in VALIDATION_VARIABLES]))
    mean_after = float(np.mean([per_variable[name]["after"] for name in VALIDATION_VARIABLES]))
    summary = {
        "schema_version": 2,
        "status": STATUS,
        "metric": "maximum signed-weighted empirical-CDF distance",
        "metric_is_standard_ks_pvalue": False,
        "per_variable": per_variable,
        "arithmetic_mean_variables": VALIDATION_VARIABLES,
        "auxiliary_variables_excluded_from_mean": AUXILIARY_VALIDATION_VARIABLES,
        "arithmetic_mean": {
            "before": mean_before,
            "after": mean_after,
            "change_before_minus_after": mean_before - mean_after,
        },
        "weights": {
            "data_signed_sweight": extended_weight_summary(data_weight),
            "mc_before": extended_weight_summary(mc_before),
            "mc_after_reweight": extended_weight_summary(mc_after),
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
        for variable in [*VALIDATION_VARIABLES, *AUXILIARY_VALIDATION_VARIABLES]:
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
        {"schema_version": 2, "status": STATUS, "variables": binning},
    )

    manifest = {
        "schema_version": 2,
        "study": "ppRef_X_reweight_transfer_closure_point_estimate",
        "status": STATUS,
        "reweight_tag": args.reweight_tag,
        "reweight_model": str((reweight_dir / source_manifest["artifacts"]["model"]).resolve()),
        "source_reweight_manifest": str(source_manifest_path.resolve()),
        "code_commit": args.code_commit,
        "inputs": {
            "data": {"path": str(Path(args.data_root).resolve()), "tree": args.data_tree, "weight": args.data_weight},
            "mc": {"path": str(Path(args.mc_root).resolve()), "tree": args.mc_tree, "weight": f"unit before; {reweight_label} after"},
        },
        "selection": selection,
        "reweight_variable_set": reweight_label,
        "reweight_variables": reweight_variables,
        "validation_variables": VALIDATION_VARIABLES,
        "auxiliary_validation_variables_excluded_from_mean": AUXILIARY_VALIDATION_VARIABLES,
        "auxiliary_validation_ranges": {
            "Btrk2dR": args.btrk2dr_range,
        },
        "binning": {
            "method": "common equal-width bins spanning selected X DATA and X MC",
            "requested_bins": args.bins,
            "histogram_quantity": "normalized density per unit variable",
            "full_selected_data_and_mc_range_included": True,
            "edges_artifact": "binning_and_histograms.json",
        },
        "weight_definitions": {
            "data": "signed signal_sWeight, unchanged",
            "mc_before": "unit weight",
            "mc_after": f"{reweight_label} model prediction multiplied by unit original weight",
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
        f"# ppRef X {reweight_label} transfer-closure point estimate",
        "",
        f"**{STATUS}.**",
        "",
        "The maximum signed-weighted empirical-CDF distance is descriptive and is not a KS p-value.",
        "No bootstrap or mass-fit/sWeight systematic variation is included, so before/after changes",
        "must not be interpreted as statistically significant improvements.",
        "",
        f"The arithmetic mean changes from {mean_before:.6g} before to {mean_after:.6g} after {reweight_label}.",
        "Per-variable changes and signed-weight stability are recorded in the JSON/CSV artifacts.",
        "The low signed-DATA effective sample size remains the principal interpretation caveat.",
    ]
    (output_dir / "interpretation.md").write_text("\n".join(interpretation) + "\n")
    print(f"Transfer closure: {output_dir}")


if __name__ == "__main__":
    main()
