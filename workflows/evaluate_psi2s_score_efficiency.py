#!/usr/bin/env python3
"""Evaluate fiducial-conditional psi(2S) score efficiencies at X working points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_extra_mc_apply_config,
    resolve_fiducial_config,
)
from utils.selection import apply_selection, selection_columns


TARGET_EFFICIENCIES = (0.2, 0.3, 0.4, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int, z: float = 1.0) -> list[float]:
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        fraction * (1.0 - fraction) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - half_width, center + half_width]


def _load_working_points(path: Path, tag: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("train_tag") != tag:
        raise ValueError(
            f"Threshold train_tag mismatch: expected {tag}, got {payload.get('train_tag')}"
        )
    by_target = {round(float(row["target_efficiency"]), 10): row for row in payload["thresholds"]}
    rows = []
    for target in TARGET_EFFICIENCIES:
        row = by_target.get(round(target, 10))
        if row is None:
            raise ValueError(f"Missing X target efficiency {target:.0%} in {path}")
        rows.append(row)
    return rows


def evaluate(tag: str) -> Path:
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    dataset_year = infer_dataset_year(tag, sample)
    if (sample, channel, dataset_year) != ("pbpb", "X", "2024"):
        raise ValueError(f"This workflow only supports PbPb24 X tags, got {tag}")

    fid_profile = infer_fid_profile(tag, sample)
    fid_expression = resolve_fiducial_config(sample, channel, fid_profile)["expression"]
    extra = resolve_extra_mc_apply_config(sample, channel, dataset_year, "psi2s")["samples"]["psi2s"]

    selected_dir = Path("output/selected") / tag
    input_path = selected_dir / "MC_psi2s_with_score.root"
    threshold_path = selected_dir / "cut_scan/weighted_signal_efficiency/thresholds.json"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not threshold_path.exists():
        raise FileNotFoundError(threshold_path)

    working_points = _load_working_points(threshold_path, tag)
    columns = list(dict.fromkeys(["Prediction", *selection_columns(fid_expression)]))
    with uproot.open(input_path) as root_file:
        tree = root_file[extra["tree"]]
        missing = [column for column in columns if column not in tree.keys()]
        if missing:
            raise ValueError(f"Missing required branches in {input_path}: {missing}")
        input_entries = int(tree.num_entries)
        frame = tree.arrays(columns, library="pd")

    fiducial = apply_selection(frame, fid_expression, f"fiducial_profiles[{fid_profile}]")
    scores = fiducial["Prediction"].to_numpy(dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite Prediction values after {fid_profile} selection")
    denominator = int(len(scores))
    if denominator == 0:
        raise ValueError(f"No psi(2S) MC entries pass {fid_profile}")

    results = []
    for row in working_points:
        threshold = float(row["score_threshold"])
        passed = int(np.count_nonzero(scores > threshold))
        efficiency = passed / denominator
        results.append(
            {
                "target_x_weighted_signal_efficiency": float(row["target_efficiency"]),
                "x_achieved_weighted_signal_efficiency": float(row["achieved_efficiency"]),
                "score_threshold": threshold,
                "comparison": "Prediction > score_threshold",
                "n_denominator": denominator,
                "n_pass": passed,
                "psi2s_score_efficiency": efficiency,
                "binomial_standard_error": math.sqrt(efficiency * (1.0 - efficiency) / denominator),
                "wilson_68pct_interval": _wilson_interval(passed, denominator),
            }
        )

    thresholds = [row["score_threshold"] for row in results]
    pass_counts = [row["n_pass"] for row in results]
    if thresholds != sorted(thresholds, reverse=True):
        raise ValueError("Score thresholds are not decreasing from 20% to 50% X efficiency")
    if pass_counts != sorted(pass_counts):
        raise ValueError("psi(2S) pass counts are not increasing from 20% to 50% X efficiency")

    output_dir = selected_dir / "psi2s_score_efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "efficiencies.json"
    payload = {
        "schema_version": 1,
        "train_tag": tag,
        "definition": "N(fiducial and Prediction > threshold) / N(fiducial)",
        "acceptance_included": False,
        "event_weights_used": False,
        "input": {
            "path": str(input_path),
            "tree": extra["tree"],
            "entries": input_entries,
            "sha256": _sha256(input_path),
        },
        "fiducial_selection": {"profile": fid_profile, "expression": fid_expression},
        "threshold_source": {"path": str(threshold_path), "sha256": _sha256(threshold_path)},
        "working_points": results,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_tags", nargs="+", help="PbPb24 X training tags")
    args = parser.parse_args()
    for tag in args.train_tags:
        path = evaluate(tag)
        print(path)


if __name__ == "__main__":
    main()
