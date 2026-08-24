import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_token_from_tag,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_fiducial_config,
    split_root_spec,
)
from utils.paths import comparison_dir, resolve_model_config_path, selected_dir
from utils.selection import apply_selection, selection_columns


def weighted_efficiency(scores, weights, cut):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores, weights = scores[valid], weights[valid]
    if weights.size == 0 or weights.sum() <= 0:
        raise ValueError("No finite positive-weight entries available")
    passed = scores > cut
    efficiency = float(weights[passed].sum() / weights.sum())
    variance = float(np.square(weights * (passed.astype(float) - efficiency)).sum() / weights.sum() ** 2)
    return efficiency, variance ** 0.5


def load_reference(tag):
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    fid_profile = infer_fid_profile(tag, sample)
    fiducial = resolve_fiducial_config(sample, channel, fid_profile)["expression"]
    with open(resolve_model_config_path(tag)) as handle:
        model_config = json.load(handle)
    weight_branch = model_config.get("efficiency_reference_weight_branch")
    if not weight_branch:
        raise ValueError(f"Tag '{tag}' has no weighted efficiency reference")
    _, tree_name = split_root_spec(model_config["efficiency_reference_signal"])
    input_path = Path(selected_dir(tag)) / "REFERENCE_MC_with_score.root"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing reference MC score file: {input_path}")
    with uproot.open(input_path) as root_file:
        tree = root_file[tree_name]
        score_branch = f"Prediction_{tag}" if f"Prediction_{tag}" in tree else "Prediction"
        columns = list(dict.fromkeys([score_branch, weight_branch] + selection_columns(fiducial)))
        frame = tree.arrays(columns, library="pd")
    frame = apply_selection(frame, fiducial, f"fiducial profile {fid_profile}")
    return {
        "tag": tag,
        "dataset": infer_dataset_token_from_tag(tag),
        "input_path": str(input_path),
        "tree": tree_name,
        "score_branch": score_branch,
        "weight_branch": weight_branch,
        "fid_profile": fid_profile,
        "fiducial_selection": fiducial,
        "entries": int(len(frame)),
        "scores": frame[score_branch].to_numpy(dtype=float),
        "weights": frame[weight_branch].to_numpy(dtype=float),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare weighted signal efficiencies at common score cuts.")
    parser.add_argument("comparison_tag")
    parser.add_argument("train_tag_a")
    parser.add_argument("train_tag_b")
    args = parser.parse_args()

    cuts = np.round(np.arange(0.05, 1.0, 0.05), 2)
    samples = [load_reference(args.train_tag_a), load_reference(args.train_tag_b)]
    if samples[0]["fiducial_selection"] != samples[1]["fiducial_selection"]:
        raise ValueError("The two tags do not resolve to the same fiducial selection")

    rows = []
    for cut in cuts:
        eff_a, err_a = weighted_efficiency(samples[0]["scores"], samples[0]["weights"], cut)
        eff_b, err_b = weighted_efficiency(samples[1]["scores"], samples[1]["weights"], cut)
        delta = eff_a - eff_b
        delta_err = float(np.hypot(err_a, err_b))
        rows.append({
            "score_cut": float(cut),
            "efficiency_a": eff_a,
            "stat_error_a": err_a,
            "efficiency_b": eff_b,
            "stat_error_b": err_b,
            "delta_a_minus_b": delta,
            "delta_stat_error": delta_err,
            "absolute_delta": abs(delta),
        })

    output_dir = Path(comparison_dir(args.comparison_tag))
    outputs = [output_dir / "score_efficiency_comparison.json", output_dir / "score_efficiency_comparison.csv", output_dir / "score_efficiency_comparison.pdf"]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite comparison outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for sample in samples:
        metadata.append({key: sample[key] for key in (
            "tag", "dataset", "input_path", "tree", "score_branch", "weight_branch",
            "fid_profile", "fiducial_selection", "entries",
        )})
    payload = {
        "schema_version": 1,
        "comparison_tag": args.comparison_tag,
        "efficiency_definition": "sum(weight for score > cut) / sum(weight)",
        "stat_error_definition": "sqrt(sum(w^2*(I-eff)^2)) / sum(w)",
        "samples": metadata,
        "rows": rows,
    }
    outputs[0].write_text(json.dumps(payload, indent=2))
    with outputs[1].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["score_cut"] for row in rows])
    fig, (ax_eff, ax_delta) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for idx, sample in enumerate(samples):
        eff = np.array([row[f"efficiency_{'a' if idx == 0 else 'b'}"] for row in rows])
        err = np.array([row[f"stat_error_{'a' if idx == 0 else 'b'}"] for row in rows])
        ax_eff.errorbar(x, eff, yerr=err, marker="o", markersize=3, capsize=2, label=sample["dataset"])
    ax_eff.set_ylabel("Weighted X efficiency")
    ax_eff.grid(alpha=0.3)
    ax_eff.legend()
    delta = np.array([row["delta_a_minus_b"] for row in rows])
    delta_err = np.array([row["delta_stat_error"] for row in rows])
    ax_delta.errorbar(x, delta, yerr=delta_err, marker="o", markersize=3, capsize=2)
    ax_delta.axhline(0.0, color="gray", linestyle="--")
    ax_delta.set_xlabel("Common XGBoost score cut")
    ax_delta.set_ylabel("Efficiency A - B")
    ax_delta.grid(alpha=0.3)
    fig.suptitle(args.comparison_tag)
    fig.tight_layout()
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[2])
    plt.close(fig)
    print(outputs[0])


if __name__ == "__main__":
    main()
