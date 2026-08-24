#!/usr/bin/env python3
"""Export the versioned PbPb24 Psi2S nominal fit-scan manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_token_from_tag,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_apply_config,
    resolve_fiducial_config,
)
from utils.paths import selected_dir


SCHEMA_VERSION = 1
CONTRACT = "pbpb24_psi2s_nominal_fit_scan"
MANIFEST_FILENAME = "fit_scan_manifest.psi2s_nominal_v1.json"
SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas/pbpb24_psi2s_nominal_fit_scan_v1.schema.json"
)
SCORE_BRANCH = "Prediction"
MASS_BRANCH = "Bmass"
MC_WEIGHT_BRANCH = "Reweight"
TARGET_EFFICIENCIES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def _to_root_expr(expression: str) -> str:
    out = str(expression).strip()
    out = out.replace("&&", " and ").replace("||", " or ")
    out = re.sub(r"\band\b", "&&", out)
    out = re.sub(r"\bor\b", "||", out)
    out = re.sub(r"\bnot\b", "!", out)
    return " ".join(out.split())


def _load_thresholds(path: Path, train_tag: str) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("train_tag") != train_tag:
        raise ValueError(
            f"Threshold train_tag mismatch: expected {train_tag}, "
            f"got {payload.get('train_tag')}"
        )
    if payload.get("weight_branch") != MC_WEIGHT_BRANCH:
        raise ValueError(
            f"Expected threshold weight branch '{MC_WEIGHT_BRANCH}', "
            f"got {payload.get('weight_branch')!r}"
        )

    by_target = {
        round(float(row["target_efficiency"]), 10): row
        for row in payload.get("thresholds", [])
    }
    rows = []
    for target in TARGET_EFFICIENCIES:
        row = by_target.get(round(target, 10))
        if row is None:
            raise ValueError(f"Missing target efficiency {target:.0%} in {path}")
        rows.append(row)

    thresholds = [float(row["score_threshold"]) for row in rows]
    if thresholds != sorted(thresholds, reverse=True):
        raise ValueError("Score thresholds must decrease from 10% to 40% efficiency")
    data_entries = [int(row["data_entries"]) for row in rows]
    if data_entries != sorted(data_entries):
        raise ValueError("Selected DATA entries must increase from 10% to 40% efficiency")
    return payload, rows


def _validate_scored_input(
    path: Path,
    tree_name: str,
    required_branches: tuple[str, ...],
) -> int:
    """Validate file/tree/branch metadata without scanning event payloads."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"Missing TTree '{tree_name}' in {path}")
        tree = root_file[tree_name]
        available = set(tree.keys())
        missing = [name for name in required_branches if name not in available]
        if missing:
            raise ValueError(f"Missing required fields in {path}:{tree_name}: {missing}")
        return int(tree.num_entries)


def build_fit_scan_manifest(
    train_tag: str,
    selected_directory: Optional[Path] = None,
) -> dict:
    sample = infer_sample_from_tag(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset = infer_dataset_token_from_tag(train_tag)
    if sample != "pbpb" or channel != "Psi2S" or dataset != "pb24":
        raise ValueError(
            f"This exporter only supports PbPb24 Psi2S tags, got '{train_tag}'"
        )

    output_directory = Path(selected_directory or selected_dir(train_tag)).resolve()
    threshold_path = (
        output_directory / "cut_scan/weighted_signal_efficiency/thresholds.json"
    ).resolve()
    data_path = (output_directory / "DATA_with_score.root").resolve()
    signal_mc_path = (output_directory / "MC_with_score.root").resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(threshold_path)

    threshold_payload, threshold_rows = _load_thresholds(threshold_path, train_tag)
    apply_config = resolve_apply_config(sample, channel, "2024")
    data_tree = apply_config["data"][0]["tree"]
    signal_mc_tree = apply_config["mc"][0]["tree"]
    data_entries = _validate_scored_input(
        data_path, data_tree, (MASS_BRANCH, SCORE_BRANCH)
    )
    signal_mc_entries = _validate_scored_input(
        signal_mc_path,
        signal_mc_tree,
        (MASS_BRANCH, SCORE_BRANCH, MC_WEIGHT_BRANCH),
    )

    fid_profile = infer_fid_profile(train_tag, sample)
    fid_expression = resolve_fiducial_config(sample, channel, fid_profile)["expression"]
    fid_root = _to_root_expr(fid_expression)

    working_points = []
    for row in threshold_rows:
        target = float(row["target_efficiency"])
        threshold = float(row["score_threshold"])
        score_cut = f"{SCORE_BRANCH} > {threshold:.17g}"
        working_points.append(
            {
                "key": f"psi2seff{int(round(target * 100)):02d}",
                "threshold": threshold,
                "target_weighted_efficiency": target,
                "achieved_weighted_efficiency": float(row["achieved_efficiency"]),
                "fiducial_score_selected_data_entries": int(row["data_entries"]),
                "selection": f"({fid_root}) && ({score_cut})",
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "train_tag": train_tag,
        "channel": channel,
        "system": "PbPb",
        "dataset": dataset,
        "path_base": "manifest_directory",
        "inputs": {
            "data": {
                "path": data_path.name,
                "tree": data_tree,
                "entries": data_entries,
                "mass_branch": MASS_BRANCH,
                "score_branch": SCORE_BRANCH,
                "event_weight": "unit",
            },
            "signal_mc": {
                "path": signal_mc_path.name,
                "tree": signal_mc_tree,
                "entries": signal_mc_entries,
                "mass_branch": MASS_BRANCH,
                "score_branch": SCORE_BRANCH,
                "event_weight_branch": MC_WEIGHT_BRANCH,
                "weight_usage": "signal_shape_and_efficiency",
            },
        },
        "score": {
            "branch": SCORE_BRANCH,
            "comparison_operator": ">",
            "threshold_boundary": "exclusive",
            "equality_passes": False,
        },
        "fiducial_selection": {
            "profile": fid_profile,
            "expression": fid_root,
        },
        "threshold_provenance": {
            "path": str(threshold_path.relative_to(output_directory)),
            "definition": threshold_payload.get("efficiency_label"),
            "event_weight_branch": MC_WEIGHT_BRANCH,
        },
        "working_points": working_points,
        "nominal_fit_contract": {
            "version": 1,
            "fit_type": "extended_unbinned",
            "mass_range_gev": [3.60, 3.80],
            "data_event_weight": "unit",
            "signal": {
                "model": "double_gaussian_mc_shape",
                "shape_source": "weighted_signal_mc",
                "event_weight_branch": MC_WEIGHT_BRANCH,
                "shared_mean": True,
                "fixed_from_mc": ["sigma1", "sigma2", "fraction"],
                "data_mean_gev": {"range": [3.6811, 3.6911]},
                "data_mc_width_scale": {"range": [0.90, 1.15]},
            },
            "background": {
                "model": "chebyshev",
                "order": 2,
                "coefficient_ranges": {"a0": [-0.8, 0.8], "a1": [-0.8, 0.8]},
                "additional_stability_models_required": False,
            },
        },
        "required_outputs": {
            "per_point_fields": [
                "point",
                "target_weighted_efficiency",
                "achieved_weighted_efficiency",
                "threshold",
                "data_entries",
                "mc_entries",
                "mc_sumw",
                "mc_sumw2",
                "mc_effective_entries",
                "yield",
                "yield_error",
                "fit_status",
                "covQual",
                "EDM",
                "mean",
                "width_scale",
                "background_parameters",
                "signal_over_background",
                "signal_over_sqrt_signal_plus_background",
                "parameter_boundary_flags",
                "artifact_paths",
            ]
        },
    }


def export_fit_scan_manifest(
    train_tag: str,
    selected_directory: Optional[Path] = None,
) -> Path:
    output_directory = Path(selected_directory or selected_dir(train_tag)).resolve()
    manifest = build_fit_scan_manifest(train_tag, output_directory)
    output_path = output_directory / MANIFEST_FILENAME
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_tag")
    args = parser.parse_args()
    print(export_fit_scan_manifest(args.train_tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
