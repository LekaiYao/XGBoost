#!/usr/bin/env python3
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
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_apply_config,
    resolve_draw_config,
    resolve_fiducial_config,
    resolve_training_config,
)
from utils.paths import selected_dir


SCHEMA_VERSION = 1
SCORE_BRANCH = "Prediction"

_FIXED_SIGNAL_REGIONS = {
    "Bu": (5.2, 5.36),
    "Bd": (5.2, 5.36),
    "Bs": (5.3, 5.46),
}


def _mass_windows_from_selection(expression: str) -> list[tuple[float, float]]:
    if not expression:
        raise ValueError("background_selection is empty; cannot infer sideband windows.")

    normalized = expression.replace("&&", " and ").replace("||", " or ")
    number = r"([0-9]*\.?[0-9]+)"
    patterns = [
        re.compile(rf"{number}\s*<\s*Bmass\s*<\s*{number}", re.IGNORECASE),
        re.compile(rf"{number}\s*>\s*Bmass\s*>\s*{number}", re.IGNORECASE),
        re.compile(rf"Bmass\s*>\s*{number}\s*and\s*Bmass\s*<\s*{number}", re.IGNORECASE),
        re.compile(rf"Bmass\s*<\s*{number}\s*and\s*Bmass\s*>\s*{number}", re.IGNORECASE),
        re.compile(
            rf"\(\s*Bmass\s*>\s*{number}\s*\)\s*and\s*"
            rf"\(\s*Bmass\s*<\s*{number}\s*\)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\(\s*Bmass\s*<\s*{number}\s*\)\s*and\s*"
            rf"\(\s*Bmass\s*>\s*{number}\s*\)",
            re.IGNORECASE,
        ),
    ]

    windows = set()
    for pattern in patterns:
        for first, second in pattern.findall(normalized):
            low, high = sorted((float(first), float(second)))
            if high > low:
                windows.add((low, high))

    if not windows:
        raise ValueError(
            f"No Bmass sideband windows found in background_selection: {expression}"
        )
    return sorted(windows)


def _suggested_signal_region(channel: str, windows: list[tuple[float, float]]) -> tuple[float, float]:
    if channel in _FIXED_SIGNAL_REGIONS:
        return _FIXED_SIGNAL_REGIONS[channel]
    if channel == "X" and len(windows) >= 2:
        low, high = windows[0][1], windows[-1][0]
        if high > low:
            return low, high
    raise ValueError(
        f"Cannot infer a suggested signal region for channel={channel}, sidebands={windows}."
    )


def _validate_root_artifact(path: Path, tree_name: str, score_branch: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing apply ROOT file: {path}")
    try:
        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                raise ValueError(f"Missing TTree '{tree_name}' in {path}")
            tree = root_file[tree_name]
            if score_branch not in tree.keys():
                raise ValueError(
                    f"Missing score branch '{score_branch}' in {path}:{tree_name}"
                )
            return {
                "file_size_bytes": path.stat().st_size,
                "entries": tree.num_entries,
            }
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read ROOT artifact {path}: {exc}") from exc


def build_manifest(train_tag: str, selected_directory: Optional[Path] = None) -> dict:
    sample = infer_sample_from_tag(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset = infer_dataset_token_from_tag(train_tag)
    dataset_year = infer_dataset_year(train_tag, sample)
    selection_profile = infer_selection_profile(train_tag, sample)
    fid_profile = infer_fid_profile(train_tag, sample)

    training_cfg = resolve_training_config(
        sample, channel, dataset_year, selection_profile
    )
    apply_cfg = resolve_apply_config(sample, channel, dataset_year)
    draw_cfg = resolve_draw_config(sample, channel, dataset_year)
    fid_cfg = resolve_fiducial_config(sample, channel, fid_profile)

    output_directory = Path(selected_directory or selected_dir(train_tag)).resolve()
    data_path = output_directory / "DATA_with_score.root"
    mc_path = output_directory / "MC_with_score.root"
    data_tree = apply_cfg["data"][0]["tree"]
    mc_tree = apply_cfg["mc"][0]["tree"]

    data_validation = _validate_root_artifact(data_path, data_tree, SCORE_BRANCH)
    mc_validation = _validate_root_artifact(mc_path, mc_tree, SCORE_BRANCH)

    sideband_windows = _mass_windows_from_selection(
        training_cfg["background_selection"]
    )
    signal_low, signal_high = _suggested_signal_region(channel, sideband_windows)
    mass_low, mass_high = draw_cfg["plot"]["mass_range"]

    return {
        "schema_version": SCHEMA_VERSION,
        "train_tag": train_tag,
        "channel": channel,
        "dataset": dataset,
        "collision_system": "pp" if sample == "pp" else "PbPb",
        "path_base": "manifest_directory",
        "score_branch": SCORE_BRANCH,
        "artifacts": {
            "data": {
                "path": data_path.name,
                "tree": data_tree,
                **data_validation,
            },
            "mc": {
                "path": mc_path.name,
                "tree": mc_tree,
                **mc_validation,
            },
        },
        "profiles": {
            "fiducial": {
                "name": fid_profile,
                "expression": fid_cfg["expression"],
            },
            "selection": {
                "name": selection_profile,
                "signal_expression": training_cfg["signal_selection"],
                "background_expression": training_cfg["background_selection"],
            },
        },
        "sideband_windows": [
            {"low": low, "high": high} for low, high in sideband_windows
        ],
        "suggested_signal_region": {
            "low": signal_low,
            "high": signal_high,
        },
        "suggested_mass_range": {
            "low": float(mass_low),
            "high": float(mass_high),
        },
        "suggested_bin_width": float(draw_cfg["plot"]["bin_width"]),
    }


def export_manifest(train_tag: str, selected_directory: Optional[Path] = None) -> Path:
    output_directory = Path(selected_directory or selected_dir(train_tag)).resolve()
    manifest = build_manifest(train_tag, output_directory)
    manifest_path = output_directory / "analysis_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate apply ROOT artifacts and export a downstream analysis manifest."
    )
    parser.add_argument("train_tag")
    args = parser.parse_args()

    manifest_path = export_manifest(args.train_tag)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
