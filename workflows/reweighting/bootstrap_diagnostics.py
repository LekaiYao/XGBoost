import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from workflows.reweighting.core import (
    load_tree_frame,
    positive_weight_tail_summary,
    predict_reweight,
    resolve_weights,
    select_frame,
    train_folding_reweighter,
    validate_columns,
    weighted_cdf_distance,
    write_json,
)

DEFAULT_TAGS = [
    "X_pp24_psi2s_R3_rw_v1",
    "X_pp24_psi2s_R4_noCos_rw_v1",
    "X_pp24_psi2s_R5_rw_v1",
    "X_pp24_psi2s_R8_rw_v1",
]
PROFILE_ORDER = ["R3", "R4_noCos", "R5", "R8"]
INTERVAL_QUANTILES = [0.025, 0.16, 0.5, 0.84, 0.975]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-tail and retraining-bootstrap diagnostics for ppRef reweighters."
    )
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument(
        "--output-dir",
        default="output/reweighting/diagnostics_pp24_psi2s_bootstrap_v1",
    )
    parser.add_argument("--replicas", type=int, default=100)
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=260729)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--code-commit",
        default=None,
        help="Git commit identifying the implementation used for the final manifest.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refresh-interpretation",
        action="store_true",
        help="Refresh tail summaries and interpretation from existing replica artifacts.",
    )
    parser.add_argument("--replicas-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_profiles(tags):
    root = Path("output/reweighting")
    profiles = {}
    reference = None
    for tag in tags:
        directory = root / tag
        manifest = json.loads((directory / "reweighting_manifest.json").read_text())
        profile = manifest["variable_set"]
        if profile in profiles:
            raise ValueError(f"Duplicate profile label: {profile}")
        profiles[profile] = {
            "tag": tag,
            "directory": directory,
            "manifest": manifest,
            "model": joblib.load(directory / manifest["artifacts"]["model"]),
        }
        if reference is None:
            reference = manifest
        else:
            for key in ("inputs", "selection", "validation_variables"):
                if manifest[key] != reference[key]:
                    raise ValueError(f"Profile {profile} has inconsistent {key}")
    missing = set(PROFILE_ORDER) - set(profiles)
    if missing:
        raise ValueError(f"Missing required profiles: {sorted(missing)}")
    return profiles, reference


def load_samples(manifest):
    inputs = manifest["inputs"]
    original = select_frame(
        load_tree_frame(inputs["original"]["path"], inputs["original"]["tree"]),
        manifest["selection"],
        "original selection",
    ).reset_index(drop=True)
    target = select_frame(
        load_tree_frame(inputs["target"]["path"], inputs["target"]["tree"]),
        manifest["selection"],
        "target selection",
    ).reset_index(drop=True)
    variables = manifest["validation_variables"]
    validate_columns(original, variables, "original sample")
    validate_columns(target, variables, "target sample")
    original_weight = resolve_weights(
        original, inputs["original"]["weight_branch"], "original sample"
    )
    target_weight = resolve_weights(
        target, inputs["target"]["weight_branch"], "target sample"
    )
    return original, target, original_weight, target_weight


def split_indices(size, fraction, seed):
    if not 0.0 < fraction < 1.0:
        raise ValueError("train-fraction must be between zero and one")
    indices = np.random.RandomState(seed).permutation(size)
    stop = int(round(fraction * size))
    stop = min(max(stop, 1), size - 1)
    return indices[:stop], indices[stop:]


def profile_parameters(profile):
    return dict(profile["manifest"]["parameters"])


def train_and_measure(
    profile,
    original_train,
    target_train,
    original_train_weight,
    target_train_weight,
    original_holdout,
    target_holdout,
    original_holdout_weight,
    target_holdout_weight,
    validation_variables,
    random_state,
):
    parameters = profile_parameters(profile)
    parameters["random_state"] = int(random_state)
    model, _ = train_folding_reweighter(
        original_train,
        target_train,
        profile["manifest"]["variables"],
        original_train_weight,
        target_train_weight,
        **parameters,
    )
    corrected = predict_reweight(
        model,
        original_holdout,
        profile["manifest"]["variables"],
        original_holdout_weight,
    )
    distances = {
        variable: weighted_cdf_distance(
            original_holdout[variable],
            target_holdout[variable],
            corrected,
            target_holdout_weight,
        )
        for variable in validation_variables
    }
    distances["mean"] = float(np.mean(list(distances.values())))
    return {
        "distances": distances,
        "holdout_reweight_tail": positive_weight_tail_summary(corrected),
    }


def bootstrap_indices(indices, rng):
    return indices[rng.randint(0, len(indices), size=len(indices))]


def make_replica(
    replica_index,
    args,
    profiles,
    original,
    target,
    original_weight,
    target_weight,
    splits,
    validation_variables,
):
    seed = args.base_seed + replica_index
    seed_sequence = np.random.SeedSequence(seed)
    generators = [np.random.RandomState(value) for value in seed_sequence.generate_state(4)]
    mc_train_idx = bootstrap_indices(splits["mc_train"], generators[0])
    data_train_idx = bootstrap_indices(splits["data_train"], generators[1])
    mc_holdout_idx = bootstrap_indices(splits["mc_holdout"], generators[2])
    data_holdout_idx = bootstrap_indices(splits["data_holdout"], generators[3])

    common = {
        "original_train": original.iloc[mc_train_idx].reset_index(drop=True),
        "target_train": target.iloc[data_train_idx].reset_index(drop=True),
        "original_train_weight": original_weight[mc_train_idx],
        "target_train_weight": target_weight[data_train_idx],
        "original_holdout": original.iloc[mc_holdout_idx].reset_index(drop=True),
        "target_holdout": target.iloc[data_holdout_idx].reset_index(drop=True),
        "original_holdout_weight": original_weight[mc_holdout_idx],
        "target_holdout_weight": target_weight[data_holdout_idx],
        "validation_variables": validation_variables,
        "random_state": seed,
    }
    results = {
        profile: train_and_measure(profiles[profile], **common)
        for profile in PROFILE_ORDER
    }
    return {
        "replica_index": int(replica_index),
        "seed": int(seed),
        "paired_profiles": True,
        "resampling": {
            "mc_train_seed": int(seed_sequence.generate_state(4)[0]),
            "data_train_seed": int(seed_sequence.generate_state(4)[1]),
            "mc_holdout_seed": int(seed_sequence.generate_state(4)[2]),
            "data_holdout_seed": int(seed_sequence.generate_state(4)[3]),
            "data_sweight_bound_to_event": True,
        },
        "profiles": results,
    }


def full_model_tail_outputs(profiles, original, original_weight, output_dir):
    weights = {}
    summaries = {}
    for profile in PROFILE_ORDER:
        entry = profiles[profile]
        values = predict_reweight(
            entry["model"], original, entry["manifest"]["variables"], original_weight
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"{profile} has non-finite or non-positive Reweight values")
        weights[profile] = values
        summaries[profile] = positive_weight_tail_summary(values)

    positive = np.concatenate(list(weights.values()))
    low = float(positive.min())
    high = float(np.nextafter(positive.max(), np.inf))
    bins = np.geomspace(low, high, 72)
    colors = ["#4477AA", "#228833", "#CC6677", "#AA3377"]
    figure, (axis, landmark_axis) = plt.subplots(
        1, 2, figsize=(12.2, 5.5), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    for profile, color in zip(PROFILE_ORDER, colors):
        values = weights[profile]
        axis.hist(
            values,
            bins=bins,
            weights=np.ones(len(values)) / len(values),
            histtype="step",
            linewidth=1.6,
            color=color,
            label=profile,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Reweight (continuous event weight)")
    axis.set_ylabel("Fraction of MC events")
    axis.set_title("Complete finite positive-weight range; no overflow clipping")
    axis.legend(frameon=False, title="Profile label")

    positions = np.arange(len(PROFILE_ORDER))
    for key, marker in (("q99", "o"), ("q995", "s"), ("q999", "^"), ("maximum", "D")):
        landmark_axis.scatter(
            [summaries[profile][key] for profile in PROFILE_ORDER],
            positions,
            marker=marker,
            s=34,
            label=key,
        )
    landmark_axis.set_xscale("log")
    landmark_axis.set_yticks(positions, labels=PROFILE_ORDER)
    landmark_axis.set_xlabel("Reweight quantile / maximum")
    landmark_axis.set_ylabel("Profile label (categorical)")
    landmark_axis.set_title("Tail landmarks")
    landmark_axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "weight_distributions_full_range.pdf")
    plt.close(figure)

    write_json(output_dir / "weight_tail_summary.json", summaries)
    with (output_dir / "weight_tail_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "profile", "q99", "q995", "q999", "maximum", "effective_sample_size",
            "top_1pct_event_count", "top_1pct_weight_fraction",
            "top_0p1pct_event_count", "top_0p1pct_weight_fraction",
        ])
        for profile in PROFILE_ORDER:
            item = summaries[profile]
            writer.writerow([
                profile, item["q99"], item["q995"], item["q999"], item["maximum"],
                item["effective_sample_size"], item["top_1pct_events"]["event_count"],
                item["top_1pct_events"]["weight_fraction"],
                item["top_0p1pct_events"]["event_count"],
                item["top_0p1pct_events"]["weight_fraction"],
            ])
    return summaries


def interval_summary(values):
    q025, q16, median, q84, q975 = np.quantile(values, INTERVAL_QUANTILES)
    return {
        "bootstrap_median": float(median),
        "central_68": [float(q16), float(q84)],
        "central_95": [float(q025), float(q975)],
    }


def summarize_replica_weight_tails(replicas):
    summary = {}
    for profile in PROFILE_ORDER:
        summary[profile] = {}
        for metric in ("maximum", "effective_sample_size"):
            values = np.array([
                replica["profiles"][profile]["holdout_reweight_tail"][metric]
                for replica in replicas
            ])
            summary[profile][metric] = interval_summary(values)
        maxima = np.array([
            replica["profiles"][profile]["holdout_reweight_tail"]["maximum"]
            for replica in replicas
        ])
        effective_sizes = np.array([
            replica["profiles"][profile]["holdout_reweight_tail"][
                "effective_sample_size"
            ]
            for replica in replicas
        ])
        summary[profile]["diagnostic_failure_rates"] = {
            "maximum_gt_100": float(np.mean(maxima > 100.0)),
            "maximum_gt_1000": float(np.mean(maxima > 1000.0)),
            "effective_sample_size_lt_4000": float(np.mean(effective_sizes < 4000.0)),
        }
    return summary


def write_replica_weight_tail_summary(output_dir, summary):
    write_json(output_dir / "bootstrap_weight_tail_summary.json", summary)
    with (output_dir / "bootstrap_weight_tail_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "profile", "metric", "bootstrap_median", "central_68_low",
            "central_68_high", "central_95_low", "central_95_high",
            "maximum_gt_100_rate", "maximum_gt_1000_rate", "neff_lt_4000_rate",
        ])
        for profile in PROFILE_ORDER:
            rates = summary[profile]["diagnostic_failure_rates"]
            for metric in ("maximum", "effective_sample_size"):
                item = summary[profile][metric]
                writer.writerow([
                    profile, metric, item["bootstrap_median"], *item["central_68"],
                    *item["central_95"], rates["maximum_gt_100"],
                    rates["maximum_gt_1000"],
                    rates["effective_sample_size_lt_4000"],
                ])


def aggregate_results(output_dir, nominal, validation_variables, requested_replicas):
    replica_paths = sorted((output_dir / "replicas").glob("replica_*.json"))
    replicas = [json.loads(path.read_text()) for path in replica_paths]
    metrics = [*validation_variables, "mean"]
    summary = {}
    for profile in PROFILE_ORDER:
        summary[profile] = {}
        for metric in metrics:
            values = np.array(
                [replica["profiles"][profile]["distances"][metric] for replica in replicas]
            )
            summary[profile][metric] = {
                "nominal": nominal[profile]["distances"][metric],
                **interval_summary(values),
            }

    wins = {profile: 0 for profile in PROFILE_ORDER}
    ties = 0
    for replica in replicas:
        means = np.array([replica["profiles"][p]["distances"]["mean"] for p in PROFILE_ORDER])
        winners = np.flatnonzero(np.isclose(means, means.min(), rtol=0.0, atol=1e-12))
        if len(winners) > 1:
            ties += 1
        for winner in winners:
            wins[PROFILE_ORDER[winner]] += 1.0 / len(winners)
    win_frequency = {
        profile: (float(wins[profile] / len(replicas)) if replicas else None)
        for profile in PROFILE_ORDER
    }

    weight_tail_summary = summarize_replica_weight_tails(replicas)
    payload = {
        "metric_note": (
            "Maximum signed-weighted empirical-CDF distances are descriptive effect sizes, "
            "not standard KS p-values."
        ),
        "requested_replicas": int(requested_replicas),
        "completed_replicas": len(replicas),
        "paired_profile_replicas": True,
        "win_frequency_smallest_mean_distance": win_frequency,
        "tie_replicas": int(ties),
        "profiles": summary,
        "bootstrap_weight_tail_summary": weight_tail_summary,
        "replicas": replicas,
    }
    write_json(output_dir / "bootstrap_results.json", payload)
    write_replica_weight_tail_summary(output_dir, weight_tail_summary)

    with (output_dir / "bootstrap_per_replica.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["replica_index", "seed", "profile", *metrics])
        for replica in replicas:
            for profile in PROFILE_ORDER:
                distances = replica["profiles"][profile]["distances"]
                writer.writerow([
                    replica["replica_index"], replica["seed"], profile,
                    *[distances[metric] for metric in metrics],
                ])

    with (output_dir / "bootstrap_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "profile", "metric", "nominal", "bootstrap_median",
            "central_68_low", "central_68_high", "central_95_low", "central_95_high",
        ])
        for profile in PROFILE_ORDER:
            for metric in metrics:
                item = summary[profile][metric]
                writer.writerow([
                    profile, metric, item["nominal"], item["bootstrap_median"],
                    *item["central_68"], *item["central_95"],
                ])

    plot_bootstrap_summary(summary, metrics, output_dir / "cdf_distance_bootstrap_summary.pdf")
    return payload


def plot_bootstrap_summary(summary, metrics, output):
    figure, axes = plt.subplots(3, 3, figsize=(12.5, 10.5))
    colors = ["#4477AA", "#228833", "#CC6677", "#AA3377"]
    x = np.arange(len(PROFILE_ORDER))
    for axis, metric in zip(axes.flat, metrics):
        for index, (profile, color) in enumerate(zip(PROFILE_ORDER, colors)):
            item = summary[profile][metric]
            median = item["bootstrap_median"]
            low68, high68 = item["central_68"]
            low95, high95 = item["central_95"]
            axis.errorbar(
                index, median,
                yerr=[[median - low95], [high95 - median]],
                fmt="none", ecolor=color, alpha=0.35, capsize=2,
            )
            axis.errorbar(
                index, median,
                yerr=[[median - low68], [high68 - median]],
                fmt="o", color=color, capsize=3,
            )
            axis.scatter(index, item["nominal"], marker="x", color="black", s=32)
        axis.set_xticks(x, PROFILE_ORDER, rotation=25)
        axis.set_title("Arithmetic mean" if metric == "mean" else metric)
        axis.set_ylabel("Max signed-weighted CDF distance")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Retraining bootstrap: median, central 68% / 95%; x = nominal")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def write_interpretation(output_dir, aggregate, tail):
    lines = [
        "# ppRef reweighting diagnostic interpretation",
        "",
        "The signed-weighted empirical-CDF distance is descriptive and is not a KS p-value.",
        "The event-level bootstrap resamples DATA and MC independently, keeps each signed sWeight",
        "bound to its DATA event, retrains every profile, and evaluates an independent holdout.",
        "It covers finite-event and reweighter-training fluctuations, but not mass-fit-model",
        "systematics in the sWeights.",
        "",
        "## Mean distance and paired winner frequency",
        "",
    ]
    for profile in PROFILE_ORDER:
        item = aggregate["profiles"][profile]["mean"]
        lines.append(
            f"- {profile}: nominal={item['nominal']:.6g}, median={item['bootstrap_median']:.6g}, "
            f"68%=[{item['central_68'][0]:.6g}, {item['central_68'][1]:.6g}], "
            f"95%=[{item['central_95'][0]:.6g}, {item['central_95'][1]:.6g}], "
            f"P(min)={aggregate['win_frequency_smallest_mean_distance'][profile]:.3f}."
        )
    lines.extend([
        "",
        "## Supported and inconclusive comparisons",
        "",
        "- R4_noCos has clearly worse closure than the other profiles; its mean distance is",
        "  separated upward and Bcos_dtheta remains strongly mismodelled.",
        "- The bootstrap does not establish a reliable closure ranking among R3, R5, and R8.",
        "  Their mean-distance intervals overlap substantially and the paired winner frequency",
        "  is not, by itself, a model-selection significance test.",
        "- R8 has the highest P(min), but this does not compensate for its nominal weight",
        "  instability (maximum weight 219.262 and substantially reduced nominal N_eff).",
        "- These results do not support the strong statement that R5 has statistically the best",
        "  KS-like closure. R5 remains a pragmatic nominal choice based on closure together with",
        "  weight stability and the previously agreed physics-variable scope.",
    ])
    lines.extend(["", "## Full-model positive-weight tails", ""])
    for profile in PROFILE_ORDER:
        item = tail[profile]
        lines.append(
            f"- {profile}: q99={item['q99']:.6g}, q99.5={item['q995']:.6g}, "
            f"q99.9={item['q999']:.6g}, max={item['maximum']:.6g}, "
            f"N_eff={item['effective_sample_size']:.1f}, "
            f"top 1% weight share={item['top_1pct_events']['weight_fraction']:.3f}, "
            f"top 0.1% share={item['top_0p1pct_events']['weight_fraction']:.3f}."
        )
    lines.extend(["", "## Bootstrap weight-tail stability", ""])
    for profile in PROFILE_ORDER:
        item = aggregate["bootstrap_weight_tail_summary"][profile]
        maximum = item["maximum"]
        neff = item["effective_sample_size"]
        rates = item["diagnostic_failure_rates"]
        lines.append(
            f"- {profile}: max weight median={maximum['bootstrap_median']:.3g}, "
            f"68%=[{maximum['central_68'][0]:.3g}, {maximum['central_68'][1]:.3g}], "
            f"95%=[{maximum['central_95'][0]:.3g}, {maximum['central_95'][1]:.3g}]; "
            f"N_eff median={neff['bootstrap_median']:.1f}, "
            f"68%=[{neff['central_68'][0]:.1f}, {neff['central_68'][1]:.1f}], "
            f"95%=[{neff['central_95'][0]:.1f}, {neff['central_95'][1]:.1f}]; "
            f"P(max>100)={rates['maximum_gt_100']:.3f}, "
            f"P(max>1000)={rates['maximum_gt_1000']:.3f}, "
            f"P(N_eff<4000)={rates['effective_sample_size_lt_4000']:.3f}."
        )
    lines.extend([
        "",
        "The failure-rate thresholds above are diagnostics, not acceptance criteria. R4_noCos",
        "shows the clearest tail instability. Rare extreme replicas also occur for R3 and R8;",
        "the nominal R8 maximum remains an independent warning that is not visible from the",
        "bootstrap median alone.",
    ])
    lines.extend([
        "",
        "## Follow-up fit-side input needed for sWeight systematics",
        "",
        "To cover mass-fit uncertainty, Analysis_CODES would need to provide event-level sWeight",
        "variations or fit-toy replicas with stable event identifiers, together with the fit-model/",
        "fit-range variation label and covariance/toy seed. This is not included here.",
    ])
    (output_dir / "interpretation.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.refresh_interpretation:
        aggregate = json.loads((output_dir / "bootstrap_results.json").read_text())
        weight_tail_summary = summarize_replica_weight_tails(aggregate["replicas"])
        aggregate["bootstrap_weight_tail_summary"] = weight_tail_summary
        write_json(output_dir / "bootstrap_results.json", aggregate)
        write_replica_weight_tail_summary(output_dir, weight_tail_summary)
        tail = json.loads((output_dir / "weight_tail_summary.json").read_text())
        write_interpretation(output_dir, aggregate, tail)
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["bootstrap_weight_tail_json"] = (
            "bootstrap_weight_tail_summary.json"
        )
        manifest["artifacts"]["bootstrap_weight_tail_csv"] = (
            "bootstrap_weight_tail_summary.csv"
        )
        write_json(manifest_path, manifest)
        print(f"Refreshed interpretation: {output_dir}")
        return
    replica_dir = output_dir / "replicas"
    replica_dir.mkdir(parents=True, exist_ok=True)
    profiles, reference = load_profiles(args.tags)
    original, target, original_weight, target_weight = load_samples(reference)
    validation_variables = reference["validation_variables"]
    mc_train, mc_holdout = split_indices(len(original), args.train_fraction, args.split_seed)
    data_train, data_holdout = split_indices(len(target), args.train_fraction, args.split_seed + 1)
    splits = {
        "mc_train": mc_train, "mc_holdout": mc_holdout,
        "data_train": data_train, "data_holdout": data_holdout,
    }

    if args.replicas_only:
        tail = None
        nominal = None
    else:
        tail = full_model_tail_outputs(profiles, original, original_weight, output_dir)
        nominal = {}
        for profile in PROFILE_ORDER:
            nominal[profile] = train_and_measure(
                profiles[profile],
                original.iloc[mc_train].reset_index(drop=True),
                target.iloc[data_train].reset_index(drop=True),
                original_weight[mc_train], target_weight[data_train],
                original.iloc[mc_holdout].reset_index(drop=True),
                target.iloc[data_holdout].reset_index(drop=True),
                original_weight[mc_holdout], target_weight[data_holdout],
                validation_variables, args.split_seed,
            )
        write_json(output_dir / "nominal_holdout.json", nominal)

    if not args.aggregate_only:
        for replica_index in range(args.replica_start, args.replica_start + args.replicas):
            path = replica_dir / f"replica_{replica_index:04d}.json"
            if args.skip_existing and path.exists():
                continue
            replica = make_replica(
                replica_index, args, profiles, original, target, original_weight,
                target_weight, splits, validation_variables,
            )
            write_json(path, replica)
            print(f"Replica {replica_index} complete", flush=True)

    if args.replicas_only:
        return

    aggregate = aggregate_results(output_dir, nominal, validation_variables, args.replicas)
    manifest = {
        "schema_version": 1,
        "study": "ppRef_psi2s_retraining_bootstrap_diagnostics",
        "profiles": {p: profiles[p]["tag"] for p in PROFILE_ORDER},
        "profile_configs": {
            profile: {
                "tag": profiles[profile]["tag"],
                "variables": profiles[profile]["manifest"]["variables"],
                "algorithm": profiles[profile]["manifest"]["algorithm"],
                "parameters": profiles[profile]["manifest"]["parameters"],
                "source_manifest": str(
                    profiles[profile]["directory"] / "reweighting_manifest.json"
                ),
            }
            for profile in PROFILE_ORDER
        },
        "code_commit": args.code_commit,
        "inputs": reference["inputs"],
        "selection": reference["selection"],
        "validation_variables": validation_variables,
        "metric": "maximum signed-weighted empirical-CDF distance",
        "metric_is_standard_ks_pvalue": False,
        "bootstrap": {
            "requested_replicas": args.replicas,
            "completed_replicas": aggregate["completed_replicas"],
            "base_seed": args.base_seed,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "strategy": (
                "fixed disjoint train/holdout partitions; paired profiles; independent with-"
                "replacement DATA and MC resampling inside both partitions; signed sWeight bound "
                "to DATA event; retrain FoldingReweighter for every profile and replica"
            ),
        },
        "fit_sweight_systematics_included": False,
        "artifacts": {
            "weight_pdf": "weight_distributions_full_range.pdf",
            "weight_tail_json": "weight_tail_summary.json",
            "weight_tail_csv": "weight_tail_summary.csv",
            "bootstrap_json": "bootstrap_results.json",
            "bootstrap_replica_csv": "bootstrap_per_replica.csv",
            "bootstrap_summary_csv": "bootstrap_summary.csv",
            "bootstrap_pdf": "cdf_distance_bootstrap_summary.pdf",
            "interpretation": "interpretation.md",
            "replica_directory": "replicas/",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_interpretation(output_dir, aggregate, tail)
    print(f"Diagnostics: {output_dir}")


if __name__ == "__main__":
    main()
