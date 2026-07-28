import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from utils.varsets import get_reweight_varset_columns
from workflows.reweighting.core import (
    load_tree_frame,
    predict_reweight,
    resolve_weights,
    select_frame,
    validate_columns,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reweighting closure diagnostics.")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--reference-tag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variable-set", default="R8")
    parser.add_argument("--sample", default="pp")
    parser.add_argument("--channel", default="X")
    return parser.parse_args()


def normalized_hist(values, weights, bins):
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    variances, _ = np.histogram(values, bins=bins, weights=np.square(weights))
    total = float(np.sum(weights))
    return counts / total, np.sqrt(variances) / abs(total)


def robust_bins(original, target, bins=35):
    combined = np.concatenate([np.asarray(original), np.asarray(target)])
    low, high = np.quantile(combined, [0.005, 0.995])
    if not high > low:
        low, high = float(np.min(combined)), float(np.max(combined))
    return np.linspace(low, high, bins + 1)


def plot_variable_closure(
    variable, original, target, original_weight, corrected_weight, target_weight, output
):
    bins = robust_bins(original[variable], target[variable])
    centers = 0.5 * (bins[:-1] + bins[1:])
    before, before_error = normalized_hist(original[variable], original_weight, bins)
    after, after_error = normalized_hist(original[variable], corrected_weight, bins)
    data, data_error = normalized_hist(target[variable], target_weight, bins)

    figure, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axis.errorbar(centers, data, yerr=data_error, fmt="o", color="black", label="DATA sWeight")
    axis.step(bins[:-1], before, where="post", color="#4477AA", label="MC before")
    axis.step(bins[:-1], after, where="post", color="#CC6677", label="MC after R5")
    axis.set_ylabel("Normalized entries")
    axis.legend(frameon=False)
    axis.set_title(f"ppRef $\\psi(2S)$ reweighting closure: {variable}")

    valid = np.abs(data) > 1e-12
    before_ratio = np.full_like(data, np.nan)
    after_ratio = np.full_like(data, np.nan)
    before_ratio[valid] = before[valid] / data[valid]
    after_ratio[valid] = after[valid] / data[valid]
    ratio_axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ratio_axis.plot(centers, before_ratio, "o", color="#4477AA", markersize=3)
    ratio_axis.plot(centers, after_ratio, "o", color="#CC6677", markersize=3)
    ratio_axis.set_ylim(0.0, 2.0)
    ratio_axis.set_ylabel("MC / DATA")
    ratio_axis.set_xlabel(variable)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_weight_comparison(weights, output):
    positive = np.concatenate([values[values > 0.0] for values in weights.values()])
    upper = float(np.quantile(positive, 0.995))
    lower = max(float(np.quantile(positive, 0.001)), 1e-4)
    bins = np.geomspace(lower, upper, 55)
    figure, axis = plt.subplots(figsize=(7.2, 5.5))
    colors = ["#4477AA", "#228833", "#CC6677", "#AA3377"]
    for (label, values), color in zip(weights.items(), colors):
        axis.hist(
            np.clip(values, lower, upper),
            bins=bins,
            weights=np.ones(len(values)) / len(values),
            histtype="step",
            linewidth=1.6,
            label=f"{label} (max={np.max(values):.1f})",
            color=color,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Reweight")
    axis.set_ylabel("Fraction of MC events")
    axis.set_title("ppRef $\\psi(2S)$ reweight distributions")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_auc_stability(reference_dir, output):
    seeds, before, after = [], [], []
    for path in sorted(reference_dir.glob("domain_classifier_holdout_seed*.json")):
        payload = json.loads(path.read_text())
        seeds.append(payload["split"]["random_state"])
        before.append(payload["holdout"]["before"]["signed_auc"])
        after.append(payload["holdout"]["after"]["signed_auc"])
    if not seeds:
        raise FileNotFoundError(f"No holdout seed JSON files in {reference_dir}")
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.axhline(0.5, color="0.4", linestyle="--", label="Random-domain limit")
    axis.plot(seeds, before, "o-", color="#4477AA", label="MC before")
    axis.plot(seeds, after, "o-", color="#CC6677", label="MC after R5")
    axis.set_xticks(seeds)
    axis.set_xlabel("Random seed")
    axis.set_ylabel("Signed holdout AUC")
    axis.set_ylim(0.45, 0.78)
    axis.set_title("R8 domain-classifier stability")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main():
    args = parse_args()
    root = Path("output/reweighting")
    reference_dir = root / args.reference_tag
    manifest = json.loads((reference_dir / "reweighting_manifest.json").read_text())
    inputs = manifest["inputs"]
    selection = manifest["selection"]
    variables = get_reweight_varset_columns(args.sample, args.variable_set, args.channel)
    original = select_frame(
        load_tree_frame(inputs["original"]["path"], inputs["original"]["tree"]),
        selection,
        "original selection",
    )
    target = select_frame(
        load_tree_frame(inputs["target"]["path"], inputs["target"]["tree"]),
        selection,
        "target selection",
    )
    validate_columns(original, variables, "original sample")
    validate_columns(target, variables, "target sample")
    original_weight = resolve_weights(
        original, inputs["original"]["weight_branch"], "original sample"
    )
    target_weight = resolve_weights(
        target, inputs["target"]["weight_branch"], "target sample"
    )

    weights = {}
    for tag in args.tags:
        tag_dir = root / tag
        tag_manifest = json.loads((tag_dir / "reweighting_manifest.json").read_text())
        model = joblib.load(tag_dir / "reweighter.pkl")
        weights[tag_manifest["variable_set"]] = predict_reweight(
            model, original, tag_manifest["variables"], original_weight
        )

    output_dir = Path(args.output_dir)
    variable_dir = output_dir / "variable_closure"
    variable_dir.mkdir(parents=True, exist_ok=True)
    nominal_weight = weights[
        json.loads((reference_dir / "reweighting_manifest.json").read_text())["variable_set"]
    ]
    for variable in variables:
        plot_variable_closure(
            variable,
            original,
            target,
            original_weight,
            nominal_weight,
            target_weight,
            variable_dir / f"{variable}_before_after_ratio.pdf",
        )
    plot_weight_comparison(weights, output_dir / "weight_distributions_R3_R4_R5_R8.pdf")
    plot_auc_stability(reference_dir, output_dir / "domain_auc_stability.pdf")
    print(f"Plots: {output_dir}")


if __name__ == "__main__":
    main()
