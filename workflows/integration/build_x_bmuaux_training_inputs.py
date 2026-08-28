#!/usr/bin/env python3
"""Build versioned identity-validated R6range5 X MC inputs with muon branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

from configs.samples import resolve_apply_config, resolve_training_reweight_config


BASELINES = {
    "pb23": {
        "year": "2023",
        "selection": "pb23_v3",
        "fiducial": "pb23_fid3",
        "output_stem": "flat_ntmix_PbPb23_MCX3872",
    },
    "pb24": {
        "year": "2024",
        "selection": "pb24_v19",
        "fiducial": "pb24_fid19",
        "output_stem": "flat_ntmix_PbPb24_MC_X3872",
    },
}
VARIANTS = {
    "bmuaux_v1": {
        "added_branches": ("Bmu1y", "Bmu2pt"),
        "output_directory": "expanded_training_v1",
        "output_suffix": "with_reweight_bmuaux.root",
        "training_reweight_profile": "rwr6range5bmuauxv1",
    },
    "bmuaux_v2": {
        "added_branches": ("Bmu1y", "Bmu2y", "Bmu1pt", "Bmu2pt"),
        "output_directory": "expanded_training_v2",
        "output_suffix": "with_reweight_muonaux.root",
        "training_reweight_profile": "rwr6range5v1",
    },
}
OUTPUT_ROOT = Path("output/reweighting/X_pp24_xsplot_R6range5_rw_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left.view(np.uint8), right.view(np.uint8))
    )


def build_one(
    raw_spec: dict,
    weighted_spec: dict,
    output_path: Path,
    added_branches,
    force: bool,
) -> dict:
    raw_path = Path(raw_spec["path"]).resolve()
    weighted_path = Path(weighted_spec["path"]).resolve()
    if raw_spec["tree"] != weighted_spec["tree"]:
        raise ValueError("Raw and weighted tree names differ")
    tree_name = raw_spec["tree"]
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")

    with uproot.open(raw_path) as raw_file, uproot.open(weighted_path) as weighted_file:
        raw = raw_file[tree_name]
        weighted = weighted_file[tree_name]
        if raw.num_entries != weighted.num_entries:
            raise ValueError("Raw/weighted entry-count mismatch")
        weighted_names = list(weighted.keys())
        identity_names = [name for name in weighted_names if name != "Reweight"]
        missing = [name for name in identity_names + list(added_branches) if name not in raw]
        if missing:
            raise ValueError(f"Raw input is missing branches: {missing}")

        weighted_arrays = weighted.arrays(weighted_names, library="np")
        raw_arrays = raw.arrays(identity_names + list(added_branches), library="np")
        for name in identity_names:
            if not _bitwise_equal(raw_arrays[name], weighted_arrays[name]):
                raise ValueError(f"Entry-order identity mismatch for branch {name}")
        payload = dict(weighted_arrays)
        for name in added_branches:
            payload[name] = raw_arrays[name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    with uproot.recreate(temporary) as output_file:
        output_file.mktree(tree_name, payload)
    temporary.replace(output_path)

    with uproot.open(output_path) as output_file:
        output = output_file[tree_name]
        if output.num_entries != len(next(iter(payload.values()))):
            raise ValueError("Expanded output entry count changed")
        output_arrays = output.arrays(list(payload), library="np")
        for name, values in payload.items():
            if not _bitwise_equal(output_arrays[name], values):
                raise ValueError(f"Expanded output changed branch {name}")

    return {
        "path": output_path.name,
        "tree": tree_name,
        "entries": int(len(next(iter(payload.values())))),
        "branches": list(payload),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
        "raw_source": {
            "path": str(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "mtime_ns": raw_path.stat().st_mtime_ns,
        },
        "weighted_source": {
            "path": str(weighted_path),
            "sha256": sha256(weighted_path),
        },
    }


def build_all(artifact_variant: str = "bmuaux_v1", force: bool = False) -> Path:
    variant = VARIANTS[artifact_variant]
    added_branches = variant["added_branches"]
    output_directory = OUTPUT_ROOT / variant["output_directory"]
    categories = {}
    for dataset, config in BASELINES.items():
        apply_config = resolve_apply_config("pbpb", "X", config["year"])
        raw_spec = apply_config["mc"][0]
        weighted = resolve_training_reweight_config(
            "pbpb", "X", config["year"], "rwr6range5v1",
            config["selection"], config["fiducial"],
        )["signal"]
        output_name = f"{config['output_stem']}_{variant['output_suffix']}"
        categories[dataset] = build_one(
            raw_spec, weighted, output_directory / output_name, added_branches, force
        )

    manifest = {
        "contract": "pbpb_x_r6range5_bmuaux_training_inputs",
        "schema_version": 1,
        "purpose": "controlled_ml_feature_ablation",
        "artifact_variant": artifact_variant,
        "added_branches": list(added_branches),
        "baseline_reweight_profile": "rwr6range5v1",
        "training_reweight_profile": variant["training_reweight_profile"],
        "invariants": {
            "selection_changed": False,
            "fiducial_selection_changed": False,
            "reweight_values_changed": False,
            "event_set_changed": False,
            "event_order_changed": False,
            "xgb_settings_changed": False,
        },
        "categories": categories,
    }
    manifest_path = output_directory / "input_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-variant", choices=tuple(VARIANTS), default="bmuaux_v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build_all(artifact_variant=args.artifact_variant, force=args.force))


if __name__ == "__main__":
    main()
