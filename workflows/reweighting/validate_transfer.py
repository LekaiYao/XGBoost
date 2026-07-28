import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from utils.varsets import get_reweight_varset_columns
from workflows.reweighting.core import (
    load_tree_frame,
    resolve_weights,
    select_frame,
    validate_columns,
    weight_summary,
    weighted_cdf_distance,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a control-channel reweighter transferred to another MC sample."
    )
    parser.add_argument("reweight_tag")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--transfer-root", required=True)
    parser.add_argument("--transfer-tree", required=True)
    parser.add_argument("--weight-branch", default="Reweight")
    parser.add_argument("--selection")
    parser.add_argument("--variable-set", default="R5")
    parser.add_argument("--validation-variable-set", default="R8")
    parser.add_argument("--sample", default="pp")
    parser.add_argument("--channel", default="X")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--robust-quantile", type=float, default=0.005)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def normalized_hist(values, weights, bins):
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    return counts / np.sum(weights)


def robust_bins(values, bins=40):
    low, high = np.quantile(values, [0.001, 0.999])
    return np.linspace(low, high, bins + 1)


def nearest_neighbor_support(source, transfer, variables, random_state):
    source_train, source_reference = train_test_split(
        np.arange(len(source)), test_size=0.5, random_state=random_state
    )
    train_values = source.iloc[source_train][variables].to_numpy(dtype=float)
    reference_values = source.iloc[source_reference][variables].to_numpy(dtype=float)
    transfer_values = transfer[variables].to_numpy(dtype=float)
    center = np.median(train_values, axis=0)
    scale = np.quantile(train_values, 0.75, axis=0) - np.quantile(
        train_values, 0.25, axis=0
    )
    scale = np.where(scale > 0.0, scale, 1.0)
    train_values = (train_values - center) / scale
    reference_values = (reference_values - center) / scale
    transfer_values = (transfer_values - center) / scale

    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(train_values)
    reference_distance = neighbors.kneighbors(reference_values, return_distance=True)[0][:, 0]
    transfer_distance = neighbors.kneighbors(transfer_values, return_distance=True)[0][:, 0]
    quantiles = {}
    for quantile in (0.95, 0.99, 0.999):
        threshold = float(np.quantile(reference_distance, quantile))
        quantiles[str(quantile)] = {
            "source_reference_threshold": threshold,
            "transfer_fraction_above": float(np.mean(transfer_distance > threshold)),
        }
    return reference_distance, transfer_distance, {
        "method": "nearest_source_event_after_source_IQR_scaling",
        "source_train_entries": len(source_train),
        "source_reference_entries": len(source_reference),
        "transfer_entries": len(transfer),
        "source_reference_median": float(np.median(reference_distance)),
        "transfer_median": float(np.median(transfer_distance)),
        "transfer_q99": float(np.quantile(transfer_distance, 0.99)),
        "thresholds": quantiles,
    }


def plot_support(reference_distance, transfer_distance, output):
    upper = np.quantile(np.concatenate([reference_distance, transfer_distance]), 0.995)
    bins = np.linspace(0.0, upper, 55)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.hist(
        reference_distance,
        bins=bins,
        weights=np.ones(len(reference_distance)) / len(reference_distance),
        histtype="step",
        linewidth=1.6,
        label=r"$\psi(2S)$ MC reference",
    )
    axis.hist(
        transfer_distance,
        bins=bins,
        weights=np.ones(len(transfer_distance)) / len(transfer_distance),
        histtype="step",
        linewidth=1.6,
        label=r"$X(3872)$ MC",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Nearest-source distance in R5 space")
    axis.set_ylabel("Fraction of events")
    axis.set_title("Control-to-signal R5 support")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_weights(weights, output):
    upper = np.quantile(weights, 0.999)
    bins = np.geomspace(max(np.min(weights), 1e-4), upper, 55)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.hist(
        np.clip(weights, bins[0], bins[-1]),
        bins=bins,
        weights=np.ones(len(weights)) / len(weights),
        histtype="step",
        linewidth=1.7,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Transferred R5 weight")
    axis.set_ylabel("Fraction of X MC events")
    axis.set_title(r"$\psi(2S)$ correction applied to $X(3872)$ MC")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_variable_shifts(transfer, weights, variables, output):
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.5))
    unit = np.ones(len(transfer))
    for axis, variable in zip(axes.flat, variables):
        bins = robust_bins(transfer[variable])
        before = normalized_hist(transfer[variable], unit, bins)
        after = normalized_hist(transfer[variable], weights, bins)
        axis.step(bins[:-1], before, where="post", label="X MC before")
        axis.step(bins[:-1], after, where="post", label="X MC after R5")
        distance = weighted_cdf_distance(
            transfer[variable], transfer[variable], unit, weights
        )
        axis.set_title(f"{variable}\nCDF shift={distance:.3f}")
        axis.set_ylabel("Normalized entries")
    axes.flat[0].legend(frameon=False)
    figure.suptitle(r"Effect of transferred $\psi(2S)$ correction on $X(3872)$ MC")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main():
    args = parse_args()
    variables = get_reweight_varset_columns(
        args.sample, args.variable_set, args.channel
    )
    validation_variables = get_reweight_varset_columns(
        args.sample, args.validation_variable_set, args.channel
    )
    all_variables = list(dict.fromkeys([*variables, *validation_variables]))
    source = select_frame(
        load_tree_frame(args.source_root, args.source_tree),
        args.selection,
        "source selection",
    )
    transfer = select_frame(
        load_tree_frame(args.transfer_root, args.transfer_tree),
        args.selection,
        "transfer selection",
    )
    validate_columns(source, all_variables, "source sample")
    validate_columns(transfer, [*all_variables, args.weight_branch], "transfer sample")
    weights = resolve_weights(transfer, args.weight_branch, "transfer sample")
    unit = np.ones(len(transfer), dtype=float)

    range_support = {}
    quantile = args.robust_quantile
    for variable in variables:
        low, high = np.quantile(source[variable], [quantile, 1.0 - quantile])
        values = transfer[variable].to_numpy(dtype=float)
        range_support[variable] = {
            "source_robust_low": float(low),
            "source_robust_high": float(high),
            "transfer_fraction_outside_robust_range": float(
                np.mean((values < low) | (values > high))
            ),
            "transfer_fraction_outside_strict_range": float(
                np.mean(
                    (values < float(source[variable].min()))
                    | (values > float(source[variable].max()))
                )
            ),
        }

    reference_distance, transfer_distance, neighbor_summary = nearest_neighbor_support(
        source, transfer, variables, args.random_state
    )
    shifts = {
        variable: weighted_cdf_distance(
            transfer[variable], transfer[variable], unit, weights
        )
        for variable in validation_variables
    }
    weight_stats = weight_summary(weights)
    weight_stats["effective_fraction"] = weight_stats["effective_sample_size"] / len(
        weights
    )

    output_dir = Path(
        args.output_dir
        or Path("output/reweighting") / args.reweight_tag / "transfer_to_x3872"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_support(
        reference_distance,
        transfer_distance,
        output_dir / "r5_support_nearest_neighbor.pdf",
    )
    plot_weights(weights, output_dir / "x3872_transferred_weights.pdf")
    plot_variable_shifts(
        transfer,
        weights,
        validation_variables,
        output_dir / "x3872_r8_before_after.pdf",
    )
    payload = {
        "schema_version": 1,
        "reweight_tag": args.reweight_tag,
        "status": "descriptive_transfer_validation",
        "selection": args.selection,
        "source": {
            "path": str(Path(args.source_root).resolve()),
            "tree": args.source_tree,
            "entries": len(source),
        },
        "transfer": {
            "path": str(Path(args.transfer_root).resolve()),
            "tree": args.transfer_tree,
            "entries": len(transfer),
            "weight_branch": args.weight_branch,
        },
        "reweight_variables": variables,
        "validation_variables": validation_variables,
        "range_support": range_support,
        "nearest_neighbor_support": neighbor_summary,
        "transferred_weights": weight_stats,
        "r8_cdf_shift_before_to_after": shifts,
        "metric_note": (
            "Support and CDF metrics are descriptive. No physics acceptance threshold "
            "is imposed by this script."
        ),
    }
    output = write_json(output_dir / "transfer_validation.json", payload)
    print(f"Transfer validation: {output}")


if __name__ == "__main__":
    main()
