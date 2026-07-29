#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from configs.samples import split_root_spec
from utils.paths import (
    ensure_dir,
    model_config_path,
    reweighting_working_points_dir,
    selected_dir,
)
from workflows.reweighting.core import load_tree_frame, select_frame, validate_columns


SCORE_BRANCH = "Prediction"


def score_cut_for_background_efficiency(scores, target_efficiency):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot define a working point from an empty background sample")
    if not np.isfinite(scores).all():
        raise ValueError("Background scores contain non-finite values")
    if not 0.0 < target_efficiency < 1.0:
        raise ValueError("Background efficiency must be between 0 and 1")
    cut = float(np.quantile(scores, 1.0 - target_efficiency, method="higher"))
    achieved = float(np.mean(scores >= cut))
    return cut, achieved


def efficiencies_at_cut(scores, cut, weights):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.shape != weights.shape:
        raise ValueError("Scores and weights must have identical shapes")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError("Scores and weights must be finite")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Efficiency weights must be non-negative with positive sum")
    passed = scores >= cut
    return {
        "unweighted": float(np.mean(passed)),
        "reweighted": float(weights[passed].sum() / weights.sum()),
        "passed_entries": int(np.count_nonzero(passed)),
        "total_entries": int(len(scores)),
        "passed_weight": float(weights[passed].sum()),
        "total_weight": float(weights.sum()),
    }


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    with path.open() as handle:
        return json.load(handle)


def _single_tree(path):
    import uproot

    with uproot.open(path) as root_file:
        trees = [key.split(";", 1)[0] for key, obj in root_file.items() if hasattr(obj, "num_entries")]
    trees = list(dict.fromkeys(trees))
    if len(trees) != 1:
        raise ValueError(f"Expected exactly one TTree in {path}, found {trees}")
    return trees[0]


def _load_tag_inputs(train_tag, signal_selection, background_selection):
    tag_dir = Path(selected_dir(train_tag))
    data_path = tag_dir / "DATA_with_score.root"
    mc_path = tag_dir / "MC_with_score.root"
    data = load_tree_frame(data_path, _single_tree(data_path))
    mc = load_tree_frame(mc_path, _single_tree(mc_path))
    validate_columns(data, [SCORE_BRANCH], f"{train_tag} DATA")
    validate_columns(mc, [SCORE_BRANCH], f"{train_tag} MC")
    return (
        select_frame(data, background_selection, f"{train_tag} DATA sideband").reset_index(drop=True),
        select_frame(mc, signal_selection, f"{train_tag} signal MC").reset_index(drop=True),
    )


def _validate_same_setup(unweighted_tag, weighted_tag):
    summaries = {
        tag: _read_json(Path(selected_dir(tag)) / "batch_apply_summary.json")
        for tag in (unweighted_tag, weighted_tag)
    }
    keys = ("input_selection", "draw_selection")
    for key in keys:
        if summaries[unweighted_tag][key] != summaries[weighted_tag][key]:
            raise ValueError(f"Tags do not share the same '{key}' configuration")
    dataset_keys = ("sample", "channel", "dataset_year", "data_input")
    for key in dataset_keys:
        if (
            summaries[unweighted_tag]["input_datasets"][key]
            != summaries[weighted_tag]["input_datasets"][key]
        ):
            raise ValueError(f"Tags do not share the same input dataset field '{key}'")
    unweighted_model = _read_json(model_config_path(unweighted_tag))
    weighted_model = _read_json(model_config_path(weighted_tag))
    if unweighted_model.get("reweight_profile") != "rw0":
        raise ValueError(f"First tag must be unweighted (rw0): {unweighted_tag}")
    weight_branch = weighted_model.get("signal_weight_branch")
    if not weight_branch:
        raise ValueError(f"Second tag has no signal weight branch: {weighted_tag}")
    if unweighted_model["input_columns"] != weighted_model["input_columns"]:
        raise ValueError("Tags do not use the same training variables")
    return summaries[unweighted_tag], weighted_model, weight_branch


def _load_and_align_weights(
    weighted_model, weight_branch, scored_mc, columns, signal_selection
):
    weight_path, weight_tree = split_root_spec(weighted_model["signal_path"])
    weighted_mc = select_frame(
        load_tree_frame(weight_path, weight_tree),
        signal_selection,
        "reweighted signal MC",
    ).reset_index(drop=True)
    validate_columns(weighted_mc, [*columns, weight_branch], "reweighted signal MC")
    if len(weighted_mc) != len(scored_mc):
        raise ValueError(
            f"Reweighted/scored MC length mismatch: {len(weighted_mc)} != {len(scored_mc)}"
        )
    mismatches = {}
    for column in columns:
        left = scored_mc[column].to_numpy(dtype=float)
        right = weighted_mc[column].to_numpy(dtype=float)
        mismatches[column] = float(np.max(np.abs(left - right)))
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=False):
            raise ValueError(f"Reweighted/scored MC event alignment failed for '{column}'")
    weights = weighted_mc[weight_branch].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Reweight branch must be finite, non-negative, and have positive sum")
    return weights, {
        "entries": int(len(weights)),
        "checked_columns": columns,
        "maximum_absolute_difference": mismatches,
        "status": "passed",
    }


def evaluate(unweighted_tag, weighted_tag, target_efficiencies, output_tag=None):
    summary, weighted_model, weight_branch = _validate_same_setup(unweighted_tag, weighted_tag)
    selection = summary["input_selection"]
    columns = weighted_model["input_columns"]
    frames = {}
    for tag in (unweighted_tag, weighted_tag):
        frames[tag] = _load_tag_inputs(
            tag,
            selection["signal_selection"],
            selection["background_selection"],
        )
    reference_mc = frames[weighted_tag][1]
    weights, alignment = _load_and_align_weights(
        weighted_model,
        weight_branch,
        reference_mc,
        columns,
        selection["signal_selection"],
    )

    results = []
    for tag in (unweighted_tag, weighted_tag):
        background, signal = frames[tag]
        for target in target_efficiencies:
            cut, achieved = score_cut_for_background_efficiency(
                background[SCORE_BRANCH].to_numpy(), target
            )
            signal_efficiencies = efficiencies_at_cut(
                signal[SCORE_BRANCH].to_numpy(), cut, weights
            )
            results.append(
                {
                    "train_tag": tag,
                    "target_background_efficiency": float(target),
                    "score_cut": cut,
                    "achieved_background_efficiency": achieved,
                    **signal_efficiencies,
                }
            )

    comparison_tag = output_tag or f"{unweighted_tag}__vs__{weighted_tag}"
    output_dir = Path(ensure_dir(reweighting_working_points_dir(comparison_tag)))
    payload = {
        "schema_version": 1,
        "score_branch": SCORE_BRANCH,
        "unweighted_tag": unweighted_tag,
        "weighted_tag": weighted_tag,
        "background_definition": {
            "source": "DATA sidebands",
            "selection_profile": selection["selection_profile"],
            "selection": selection["background_selection"],
        },
        "signal_mc_definition": {
            "selection": selection["signal_selection"],
            "weight_profile": weighted_model["reweight_profile"],
            "weight_branch": weight_branch,
            "weight_source": weighted_model["signal_path"],
        },
        "event_alignment": alignment,
        "working_points": results,
    }
    json_path = output_dir / "working_points.json"
    csv_path = output_dir / "working_points.csv"
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path, payload


def main():
    parser = argparse.ArgumentParser(
        description="Compare unweighted and reweighted models at equal DATA-sideband efficiencies."
    )
    parser.add_argument("unweighted_tag")
    parser.add_argument("weighted_tag")
    parser.add_argument(
        "--background-efficiencies",
        type=float,
        nargs="+",
        default=[0.10, 0.03, 0.01],
    )
    parser.add_argument("--output-tag")
    args = parser.parse_args()
    json_path, csv_path, payload = evaluate(
        args.unweighted_tag,
        args.weighted_tag,
        args.background_efficiencies,
        args.output_tag,
    )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    for row in payload["working_points"]:
        print(
            f"{row['train_tag']} bkg={row['achieved_background_efficiency']:.5f} "
            f"cut={row['score_cut']:.6f} "
            f"eff(unweighted)={row['unweighted']:.5f} "
            f"eff(reweighted)={row['reweighted']:.5f}"
        )


if __name__ == "__main__":
    main()
