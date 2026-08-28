#!/usr/bin/env python3
"""Export the PbPb23+PbPb24 X MC-shape simultaneous-fit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from configs.year_pairings import resolve_year_pairing
from utils.paths import resolve_model_config_path, selected_dir


SCHEMA_VERSION = 2
CONTRACT = "pbpb_x_simultaneous_year_mc_shape_nominal_fit_scan"
MANIFEST_FILENAME = "fit_scan_manifest.pb23_pb24_simultaneous_mc_shape_nominal_v2.json"
SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas/pbpb_x_simultaneous_year_mc_shape_nominal_fit_scan_v2.schema.json"
)
SCORE_BRANCH = "Prediction"
MASS_BRANCH = "Bmass"
MC_WEIGHT_BRANCH = "Reweight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_root_expr(expression: str) -> str:
    out = str(expression).strip().replace("&&", " and ").replace("||", " or ")
    out = re.sub(r"\band\b", "&&", out)
    out = re.sub(r"\bor\b", "||", out)
    out = re.sub(r"\bnot\b", "!", out)
    return " ".join(out.split())


def _validate_scored_input(
    path: Path,
    tree_name: str,
    required_branches: tuple[str, ...],
) -> int:
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


def _load_category(
    tag: str,
    dataset: str,
    manifest_directory: Path,
    targets: list[float],
) -> tuple[dict, dict]:
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    if sample != "pbpb" or channel != "X" or infer_dataset_token_from_tag(tag) != dataset:
        raise ValueError(f"Expected PbPb X {dataset} tag, got '{tag}'")

    year = "2023" if dataset == "pb23" else "2024"
    output_directory = Path(selected_dir(tag)).resolve()
    apply_config = resolve_apply_config(sample, channel, year)
    data_path = output_directory / "DATA_with_score.root"
    signal_mc_path = output_directory / "MC_with_score.root"
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

    fid_profile = infer_fid_profile(tag, sample)
    fid_expression = resolve_fiducial_config(sample, channel, fid_profile)["expression"]
    threshold_path = output_directory / "cut_scan/weighted_signal_efficiency/thresholds.json"
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    if payload.get("train_tag") != tag or payload.get("weight_branch") != MC_WEIGHT_BRANCH:
        raise ValueError(f"Invalid weighted-efficiency threshold provenance for '{tag}'")
    by_target = {
        round(float(row["target_efficiency"]), 10): row
        for row in payload.get("thresholds", [])
    }
    rows = {}
    for target in targets:
        row = by_target.get(round(target, 10))
        if row is None:
            raise ValueError(f"Missing target efficiency {target:.0%} for '{tag}'")
        rows[round(target, 10)] = row

    model_config = json.loads(
        Path(resolve_model_config_path(tag)).read_text(encoding="utf-8")
    )
    category = {
        "dataset": dataset,
        "train_tag": tag,
        "data": {
            "path": os.path.relpath(data_path, manifest_directory),
            "tree": data_tree,
            "entries": data_entries,
            "mass_branch": MASS_BRANCH,
            "score_branch": SCORE_BRANCH,
            "event_weight": "unit",
        },
        "signal_mc": {
            "path": os.path.relpath(signal_mc_path, manifest_directory),
            "tree": signal_mc_tree,
            "entries": signal_mc_entries,
            "mass_branch": MASS_BRANCH,
            "score_branch": SCORE_BRANCH,
            "event_weight_branch": MC_WEIGHT_BRANCH,
            "weight_usage": "signal_shape_and_efficiency",
        },
        "score": {
            "branch": SCORE_BRANCH,
            "comparison_operator": ">",
            "threshold_boundary": "exclusive",
            "equality_passes": False,
        },
        "fiducial_selection": {
            "profile": fid_profile,
            "expression": _to_root_expr(fid_expression),
        },
        "threshold_provenance": {
            "path": os.path.relpath(threshold_path, manifest_directory),
            "sha256": _sha256(threshold_path),
            "definition": payload.get("efficiency_label"),
            "event_weight_branch": MC_WEIGHT_BRANCH,
        },
    }
    evidence = {
        "fiducial_expression": fid_expression,
        "input_columns": model_config["input_columns"],
        "reweight_profile": model_config["reweight_profile"],
        "fixed_params": {
            key: value
            for key, value in model_config["model_params"].items()
            if key != "scale_pos_weight"
        },
        "rows": rows,
    }
    return category, evidence


def build_manifest(
    anchor_train_tag: str,
    selected_directory: Optional[Path] = None,
) -> dict:
    pairing = resolve_year_pairing(anchor_train_tag)
    manifest_directory = Path(
        selected_directory or selected_dir(anchor_train_tag)
    ).resolve()
    targets = [float(value) for value in pairing["fit_scan_efficiencies"]]
    categories, evidence = {}, {}
    for dataset in ("pb23", "pb24"):
        categories[dataset], evidence[dataset] = _load_category(
            pairing["tags"][dataset], dataset, manifest_directory, targets
        )
    for key in ("fiducial_expression", "input_columns", "reweight_profile", "fixed_params"):
        if evidence["pb23"][key] != evidence["pb24"][key]:
            raise ValueError(f"Year-pair compatibility mismatch for {key}")

    working_points = []
    for target in targets:
        point_categories = {}
        for dataset in ("pb23", "pb24"):
            row = evidence[dataset]["rows"][round(target, 10)]
            threshold = float(row["score_threshold"])
            fid = categories[dataset]["fiducial_selection"]["expression"]
            point_categories[dataset] = {
                "threshold": threshold,
                "achieved_weighted_efficiency": float(row["achieved_efficiency"]),
                "selected_data_entries": int(row["data_entries"]),
                "selection": f"({fid}) && ({SCORE_BRANCH} > {threshold:.17g})",
            }
        working_points.append(
            {
                "key": f"xeff{int(round(target * 100)):02d}",
                "target_weighted_efficiency": target,
                "categories": point_categories,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "anchor_train_tag": anchor_train_tag,
        "channel": "X",
        "system": "PbPb",
        "path_base": "manifest_directory",
        "pairing": {
            "anchor_dataset": pairing["anchor_dataset"],
            "selection_policy": pairing["selection_policy"],
            "categories": categories,
        },
        "working_points": working_points,
        "nominal_fit_contract": {
            "version": 2,
            "fit_type": "simultaneous_extended_unbinned",
            "mass_range_gev": [3.80, 3.94],
            "data_event_weight": "unit",
            "fit_sequence": [
                "fit_weighted_signal_mc_independently_by_category",
                "fit_data_simultaneously_with_fixed_category_mc_shape",
            ],
            "shared_parameters": ["data_signal_mean"],
            "category_specific_parameters": [
                "mc_signal_mean", "mc_sigma1", "mc_sigma2", "mc_fraction",
                "data_mc_width_scale", "signal_yield", "background_yield",
                "background_coefficients",
            ],
            "signal_mc": {
                "model": "common_mean_double_gaussian",
                "performed_independently_by_category": True,
                "event_weight_branch": MC_WEIGHT_BRANCH,
                "mean_gev": {"initial": 3.87169, "range": [3.86669, 3.87669]},
                "sigma1_gev": {"range": [0.001, 0.1]},
                "sigma2_gev": {"range": [0.001, 0.1]},
                "fraction": {"range": [0.01, 1.0]},
                "parameters_transferred_to_data": ["sigma1", "sigma2", "fraction"],
            },
            "data_fit": {
                "shared_mean_across_categories": True,
                "fixed_from_category_mc": ["sigma1", "sigma2", "fraction"],
                "mean_gev": {"initial": 3.87169, "range": [3.86669, 3.87669]},
                "category_width_scale": {"range": [0.90, 1.50]},
                "signal_yields": "independent_nonnegative_by_category",
            },
            "background": {
                "model": "chebyshev", "order": 2,
                "parameters": "independent_by_category",
            },
            "execution_policy": {
                "fit_only": True,
                "significance_calibration_authorized": False,
                "working_point_selection_authorized": False,
            },
        },
        "required_outputs": {
            "per_point_fields": [
                "point", "fit_status", "covQual", "EDM", "shared_data_mean",
                "pb23_mc_shape_parameters", "pb23_mc_fit_quality", "pb23_width_scale",
                "pb23_yield", "pb23_yield_error", "pb23_background_parameters",
                "pb24_mc_shape_parameters", "pb24_mc_fit_quality", "pb24_width_scale",
                "pb24_yield", "pb24_yield_error", "pb24_background_parameters",
                "parameter_boundary_flags", "artifact_paths",
            ]
        },
    }


def export_manifest(
    anchor_train_tag: str,
    selected_directory: Optional[Path] = None,
) -> Path:
    output_directory = Path(
        selected_directory or selected_dir(anchor_train_tag)
    ).resolve()
    manifest = build_manifest(anchor_train_tag, output_directory)
    output_path = output_directory / MANIFEST_FILENAME
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    hash_path = output_path.with_suffix(output_path.suffix + ".sha256")
    hash_path.write_text(f"{_sha256(output_path)}  {output_path.name}\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchor_train_tag")
    args = parser.parse_args()
    print(export_manifest(args.anchor_train_tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
