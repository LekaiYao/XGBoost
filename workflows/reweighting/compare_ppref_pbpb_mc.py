import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs.samples import resolve_training_config
from workflows.reweighting.core import (
    load_tree_frame,
    select_frame,
    validate_columns,
    weight_summary,
    weighted_cdf_distance,
    write_json,
)
from workflows.reweighting.validate_x_splot_transfer_closure import (
    AUXILIARY_VALIDATION_VARIABLES,
    VALIDATION_VARIABLES,
    common_equal_width_bins,
)


STATUS = "preliminary - fast transfer test - uncertainty not evaluated"
COMMON_FIDUCIAL = (
    "Bpt > 10 and Bpt < 50 and abs(By) < 1.6 and BQvalue < 0.15"
)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare unweighted ppRef X MC with unweighted PbPb24 X MC."
    )
    parser.add_argument("--reweight-tag", default="X_pp24_xsplot_R6range2_rw_v1")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir")
    parser.add_argument("--common-fiducial", default=COMMON_FIDUCIAL)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def package_versions():
    packages = {}
    for name in ["numpy", "pandas", "uproot", "matplotlib"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    return {"python": platform.python_version(), "packages": packages}


def normalized_histogram(values, weights, bins):
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    total = float(np.sum(weights))
    return counts / (total * np.diff(bins))


def plot_variable(variable, pp, pb, pp_weight, pb_weight, bins, distance, output):
    pp_density = normalized_histogram(pp[variable], pp_weight, bins)
    pb_density = normalized_histogram(pb[variable], pb_weight, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    figure, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(7.4, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.05},
    )
    axis.stairs(pp_density, bins, color="#CC6677", label="ppRef MC unweighted")
    axis.stairs(pb_density, bins, color="#4477AA", label="PbPb24 MC unweighted")
    axis.set_ylabel("Normalized density")
    axis.set_title(f"ppRef vs PbPb24 X MC: {variable}\nCDF distance {distance:.3f}")
    axis.text(
        0.02, 0.95, STATUS, transform=axis.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
    )
    axis.legend(frameon=False)
    valid = pb_density > np.finfo(float).eps * max(1.0, float(np.max(np.abs(pb_density))))
    ratio = np.full_like(pp_density, np.nan)
    ratio[valid] = pp_density[valid] / pb_density[valid]
    ratio_axis.scatter(centers[valid], ratio[valid], s=13, color="#AA3377")
    ratio_axis.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ratio_axis.set_ylabel("ppRef/PbPb")
    ratio_axis.set_xlabel(variable)
    ratio_axis.text(
        0.02, 0.08, f"undefined bins: {int(np.count_nonzero(~valid))}",
        transform=ratio_axis.transAxes, fontsize=8,
    )
    figure.savefig(output)
    plt.close(figure)
    return {
        "edges": bins.tolist(),
        "ppref_unweighted_density": pp_density.tolist(),
        "pbpb_unweighted_density": pb_density.tolist(),
        "undefined_ratio_bins": int(np.count_nonzero(~valid)),
    }


def main():
    args = parse_args()
    reweight_dir = Path("output/reweighting") / args.reweight_tag
    source_manifest = json.loads((reweight_dir / "reweighting_manifest.json").read_text())
    controlled_support = source_manifest.get("selection")
    if not controlled_support:
        raise ValueError("The source reweighter manifest has no controlled-support selection")
    scenarios = {
        "common_fiducial_baseline": args.common_fiducial,
        "common_fiducial_in_reweight_support": controlled_support,
    }
    variables = [*VALIDATION_VARIABLES, *AUXILIARY_VALIDATION_VARIABLES]
    required = list(dict.fromkeys([
        *variables, *source_manifest["variables"], "Bpt", "By", "BQvalue"
    ]))
    pp_spec = resolve_training_config("pp", "X", "2024", "pp24_v3")["signal"]
    pb_spec = resolve_training_config("pbpb", "X", "2024", "pb24_v1")["signal"]
    pp_all = load_tree_frame(pp_spec["path"], pp_spec["tree"], columns=required)
    pb_all = load_tree_frame(pb_spec["path"], pb_spec["tree"], columns=required)
    validate_columns(pp_all, required, "ppRef X MC")
    validate_columns(pb_all, required, "PbPb24 X MC")
    output_root = Path(
        args.output_dir or reweight_dir / "ppref_pbpb_mc_comparison_v1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for scenario, selection in scenarios.items():
        pp = select_frame(pp_all, selection, f"{scenario} ppRef selection")
        pb = select_frame(pb_all, selection, f"{scenario} PbPb selection")
        pp_weight = np.ones(len(pp), dtype=float)
        pb_weight = np.ones(len(pb), dtype=float)
        scenario_dir = output_root / scenario
        plot_dir = scenario_dir / "variables"
        plot_dir.mkdir(parents=True, exist_ok=True)
        per_variable = {}
        histograms = {}
        for variable in variables:
            bins = common_equal_width_bins(pp[variable], pb[variable], count=args.bins)
            distance = weighted_cdf_distance(
                pp[variable], pb[variable], pp_weight, pb_weight
            )
            per_variable[variable] = {"cdf_distance": distance}
            histograms[variable] = plot_variable(
                variable, pp, pb, pp_weight, pb_weight, bins, distance,
                plot_dir / f"{variable}_ppref_vs_pbpb_unweighted.pdf",
            )
        support_pp = select_frame(pp, controlled_support, f"{scenario} ppRef support")
        support_pb = select_frame(pb, controlled_support, f"{scenario} PbPb support")
        summary = {
            "schema_version": 1,
            "status": STATUS,
            "scenario": scenario,
            "selection": selection,
            "comparison": "unweighted ppRef X MC vs unweighted PbPb24 X MC",
            "metric": "maximum empirical-CDF distance; not a KS p-value",
            "entries": {"ppref": len(pp), "pbpb": len(pb)},
            "reweight_support_fraction": {
                "ppref": len(support_pp) / len(pp),
                "pbpb": len(support_pb) / len(pb),
            },
            "weights": {
                "ppref_unweighted": weight_summary(pp_weight),
                "pbpb_unweighted": weight_summary(pb_weight),
            },
            "per_variable": per_variable,
            "arithmetic_mean_8_variables": float(np.mean([
                per_variable[name]["cdf_distance"] for name in VALIDATION_VARIABLES
            ])),
            "auxiliary_variables_excluded_from_mean": AUXILIARY_VALIDATION_VARIABLES,
        }
        write_json(scenario_dir / "comparison_summary.json", summary)
        write_json(scenario_dir / "binning_and_histograms.json", {"variables": histograms})
        summaries[scenario] = summary
    manifest = {
        "schema_version": 1,
        "status": STATUS,
        "study": "unweighted_ppRef_MC_vs_unweighted_PbPb24_MC",
        "code_commit": args.code_commit,
        "source_reweight_manifest": str((reweight_dir / "reweighting_manifest.json").resolve()),
        "inputs": {"ppref": pp_spec, "pbpb": pb_spec},
        "variables": variables,
        "reweight_variables": source_manifest["variables"],
        "scenarios": scenarios,
        "interpretation": {
            "common_fiducial_baseline": "unweighted baseline without reweighter controlled-support cuts",
            "common_fiducial_in_reweight_support": "unweighted baseline inside the source manifest controlled support",
        },
        "software": package_versions(),
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"ppRef/PbPb MC comparison: {output_root}")


if __name__ == "__main__":
    main()
