#!/usr/bin/env python3
"""Build and validate versioned PbPb23+PbPb24 Psi2S closure25 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import uproot

from configs.samples import resolve_apply_config
from configs.year_pairings import resolve_year_pairing
from utils.paths import selected_dir


SCHEMA_VERSION = 1
CONTRACT = "pbpb_psi2s_expanded_closure_inputs"
ARTIFACT_VERSION = "closure25_v1"
MANIFEST_FILENAME = "closure_input_manifest.pb23_pb24_psi2s_closure25_v1.json"
SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas/pbpb_psi2s_expanded_closure_inputs_v1.schema.json"
)
TREE_NAME = "ntmix_PSI2S"
FORMAL_FIT_MANIFEST = "fit_scan_manifest.pb23_pb24_psi2s_simultaneous_v1.json"

CURRENT_BRANCHES = (
    "Bmass", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb",
    "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt",
    "Bpt", "By", "BQvalue", "Reweight", "Prediction",
)
ADDED_BRANCHES = (
    "BujvProb", "Balpha", "Btrk2Eta", "Btrk1Eta", "Btrk1Phi",
    "Btrk2Phi", "Bmu1y", "Bmu2y", "Bmu1pt", "Bmu2pt",
)
EXPANDED_BRANCHES = CURRENT_BRANCHES + ADDED_BRANCHES
RAW_IDENTITY_BRANCHES = tuple(
    name for name in CURRENT_BRANCHES if name not in ("Reweight", "Prediction")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_metadata(path: Path, tree_name: str) -> dict:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"Missing TTree '{tree_name}' in {path}")
        tree = root_file[tree_name]
        return {
            "entries": int(tree.num_entries),
            "branches": [
                {
                    "name": name,
                    "root_typename": tree[name].typename,
                    "numpy_dtype": str(tree[name].interpretation.numpy_dtype),
                }
                for name in tree.keys()
            ],
        }


def build_expanded_mc(
    raw_source: Path,
    current_scored: Path,
    output_path: Path,
    tree_name: str = TREE_NAME,
) -> dict:
    """Create one expanded ROOT after strict entry-order identity validation."""
    raw_source = Path(raw_source).resolve()
    current_scored = Path(current_scored).resolve()
    output_path = Path(output_path).resolve()
    if output_path in (raw_source, current_scored):
        raise ValueError("Expanded output must not overwrite an input ROOT")
    for path in (raw_source, current_scored):
        if not path.is_file():
            raise FileNotFoundError(path)

    staging_root = Path("/tmp/leyao")
    staging_root.mkdir(parents=True, exist_ok=True)
    raw_metadata = _tree_metadata(raw_source, tree_name)
    current_metadata = _tree_metadata(current_scored, tree_name)
    if raw_metadata["entries"] != current_metadata["entries"]:
        raise ValueError(
            f"Entry-count mismatch: raw={raw_metadata['entries']}, "
            f"current={current_metadata['entries']}"
        )
    raw_names = {row["name"] for row in raw_metadata["branches"]}
    current_names = {row["name"] for row in current_metadata["branches"]}
    missing_raw = [name for name in RAW_IDENTITY_BRANCHES + ADDED_BRANCHES if name not in raw_names]
    missing_current = [name for name in CURRENT_BRANCHES if name not in current_names]
    if missing_raw or missing_current:
        raise ValueError(
            f"Missing source branches: raw={missing_raw}, current={missing_current}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    macro = Path(__file__).resolve().with_name("BuildPsi2SExpandedClosure.C")
    with tempfile.TemporaryDirectory(prefix="psi2s_closure25_", dir=staging_root) as tmp:
        staged_raw = Path(tmp) / raw_source.name
        shutil.copy2(raw_source, staged_raw)
        raw_sha256 = sha256(staged_raw)
        for path in (staged_raw, current_scored, temporary_path):
            if '"' in str(path):
                raise ValueError(f'Unsupported quote in ROOT path: {path}')
        expression = (
            f'{macro}("{staged_raw}","{current_scored}",'
            f'"{temporary_path}","{tree_name}")'
        )
        completed = subprocess.run(
            ["root", "-l", "-b", "-q", expression],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0 or "EXPANDED_CLOSURE_OK" not in completed.stdout:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(
                "ROOT strict identity/copy validation failed:\n" + completed.stdout
            )
    temporary_path.replace(output_path)

    metadata = _tree_metadata(output_path, tree_name)
    actual_names = tuple(row["name"] for row in metadata["branches"])
    if set(actual_names) != set(EXPANDED_BRANCHES) or len(actual_names) != 25:
        raise ValueError(f"Expanded branch set mismatch: {actual_names}")
    entries = current_metadata["entries"]
    if metadata["entries"] != entries:
        raise ValueError("Expanded output entry count changed")

    return {
        "path": str(output_path),
        "tree": tree_name,
        "entries": entries,
        "branches": metadata["branches"],
        "sha256": sha256(output_path),
        "size_bytes": output_path.stat().st_size,
        "provenance": {
            "raw_source": {
                "path": str(raw_source),
                "sha256": raw_sha256,
                "entries": raw_metadata["entries"],
            },
            "current_scored": {
                "path": str(current_scored),
                "sha256": sha256(current_scored),
                "entries": entries,
            },
            "identity_validation": {
                "method": "strict_dtype_shape_and_value_equality_by_entry",
                "implementation": "ROOT_CXX_bitwise_scalar_comparison",
                "raw_identity_branches": list(RAW_IDENTITY_BRANCHES),
                "current_branches_preserved": list(CURRENT_BRANCHES),
                "raw_branches_added": list(ADDED_BRANCHES),
                "prediction_recomputed": False,
                "reweight_recomputed": False,
                "entry_order_changed": False,
            },
        },
    }


def _relative_spec(spec: dict, manifest_directory: Path) -> dict:
    out = dict(spec)
    out["path"] = os.path.relpath(Path(spec["path"]), manifest_directory)
    return out


def build_all(anchor_train_tag: str) -> Path:
    pairing = resolve_year_pairing(anchor_train_tag)
    if not anchor_train_tag.startswith("Psi2S_pb23_"):
        raise ValueError("Expected a PbPb23 Psi2S anchor tag")
    anchor_directory = Path(selected_dir(anchor_train_tag)).resolve()
    output_directory = anchor_directory / ARTIFACT_VERSION
    output_directory.mkdir(parents=True, exist_ok=True)
    formal_manifest = anchor_directory / FORMAL_FIT_MANIFEST
    if not formal_manifest.is_file():
        raise FileNotFoundError(formal_manifest)
    formal_payload = json.loads(formal_manifest.read_text(encoding="utf-8"))
    if formal_payload.get("contract") != "pbpb_psi2s_simultaneous_year_fit_scan":
        raise ValueError("Unexpected formal Psi2S simultaneous-fit manifest contract")

    categories = {}
    for dataset in ("pb23", "pb24"):
        tag = pairing["tags"][dataset]
        year = "2023" if dataset == "pb23" else "2024"
        selected = Path(selected_dir(tag)).resolve()
        apply_config = resolve_apply_config("pbpb", "Psi2S", year)
        raw_source = Path(apply_config["mc"][0]["path"]).resolve()
        current_scored = selected / "MC_with_score.root"
        output_path = output_directory / f"{tag}_MC_with_score_{ARTIFACT_VERSION}.root"
        expanded = build_expanded_mc(raw_source, current_scored, output_path)

        data_path = selected / "DATA_with_score.root"
        data_tree = apply_config["data"][0]["tree"]
        data_meta = _tree_metadata(data_path, data_tree)
        data_names = {row["name"] for row in data_meta["branches"]}
        missing_data = [name for name in ADDED_BRANCHES if name not in data_names]
        if missing_data:
            raise ValueError(f"Missing expanded DATA branches for {dataset}: {missing_data}")
        categories[dataset] = {
            "dataset": dataset,
            "train_tag": tag,
            "data": {
                "path": os.path.relpath(data_path, output_directory),
                "tree": data_tree,
                "entries": data_meta["entries"],
                "size_bytes": data_path.stat().st_size,
                "mtime_ns": data_path.stat().st_mtime_ns,
                "source_role": "existing_formal_scored_data_unchanged",
            },
            "expanded_signal_mc": _relative_spec(expanded, output_directory),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "artifact_version": ARTIFACT_VERSION,
        "anchor_train_tag": anchor_train_tag,
        "channel": "Psi2S",
        "system": "PbPb",
        "path_base": "manifest_directory",
        "formal_fit_manifest": {
            "path": os.path.relpath(formal_manifest, output_directory),
            "sha256": sha256(formal_manifest),
            "contract": formal_payload["contract"],
            "schema_version": formal_payload["schema_version"],
        },
        "categories": categories,
        "expanded_branch_names": list(EXPANDED_BRANCHES),
        "added_branch_names": list(ADDED_BRANCHES),
        "artifact_semantics": {
            "purpose": "expanded_signal_mc_branches_for_closure_only",
            "training_changed": False,
            "score_recomputed": False,
            "reweight_recomputed": False,
            "selection_changed": False,
            "event_set_changed": False,
            "event_order_changed": False,
            "triggers_downstream_execution": False,
        },
    }
    manifest_path = output_directory / MANIFEST_FILENAME
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    return manifest_path


def validate_delivery(manifest_path: Path) -> dict:
    """Fail closed on the delivered manifest, hashes, ROOT metadata, and DATA refs."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported expanded-closure manifest contract/schema")
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unexpected expanded-closure artifact version")
    if tuple(manifest.get("expanded_branch_names", ())) != EXPANDED_BRANCHES:
        raise ValueError("Manifest expanded branch list mismatch")
    if tuple(manifest.get("added_branch_names", ())) != ADDED_BRANCHES:
        raise ValueError("Manifest added branch list mismatch")
    if set(manifest.get("categories", {})) != {"pb23", "pb24"}:
        raise ValueError("Manifest categories must be exactly pb23 and pb24")

    formal = manifest["formal_fit_manifest"]
    formal_path = (manifest_path.parent / formal["path"]).resolve()
    if sha256(formal_path) != formal["sha256"]:
        raise ValueError("Formal fit manifest SHA256 mismatch")

    result = {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "categories": {}}
    for dataset, category in manifest["categories"].items():
        if category.get("dataset") != dataset:
            raise ValueError(f"{dataset} category dataset label mismatch")
        data = category["data"]
        data_path = (manifest_path.parent / data["path"]).resolve()
        if data_path.stat().st_size != data["size_bytes"] or data_path.stat().st_mtime_ns != data["mtime_ns"]:
            raise ValueError(f"{dataset} DATA metadata mismatch")
        data_meta = _tree_metadata(data_path, data["tree"])
        if data_meta["entries"] != data["entries"]:
            raise ValueError(f"{dataset} DATA entries mismatch")
        data_names = {row["name"] for row in data_meta["branches"]}
        if any(name not in data_names for name in ADDED_BRANCHES):
            raise ValueError(f"{dataset} DATA lacks expanded closure branches")

        expanded = category["expanded_signal_mc"]
        expanded_path = (manifest_path.parent / expanded["path"]).resolve()
        if sha256(expanded_path) != expanded["sha256"]:
            raise ValueError(f"{dataset} expanded MC SHA256 mismatch")
        metadata = _tree_metadata(expanded_path, expanded["tree"])
        if metadata["entries"] != expanded["entries"]:
            raise ValueError(f"{dataset} expanded MC entries mismatch")
        if metadata["branches"] != expanded["branches"]:
            raise ValueError(f"{dataset} expanded MC branch metadata mismatch")
        if tuple(row["name"] for row in metadata["branches"]) != EXPANDED_BRANCHES:
            raise ValueError(f"{dataset} expanded MC branch order/set mismatch")

        provenance = expanded["provenance"]
        current = provenance["current_scored"]
        current_path = Path(current["path"])
        if sha256(current_path) != current["sha256"]:
            raise ValueError(f"{dataset} current scored MC SHA256 mismatch")
        if _tree_metadata(current_path, expanded["tree"])["entries"] != current["entries"]:
            raise ValueError(f"{dataset} current scored MC entries mismatch")
        result["categories"][dataset] = {
            "data_entries": data["entries"],
            "expanded_mc_entries": expanded["entries"],
            "expanded_mc_sha256": expanded["sha256"],
            "branches": len(metadata["branches"]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchor_train_tag", nargs="?")
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args()
    if args.validate_only is not None:
        print(json.dumps(validate_delivery(args.validate_only), indent=2, sort_keys=True))
        return 0
    if not args.anchor_train_tag:
        parser.error("anchor_train_tag is required unless --validate-only is used")
    print(build_all(args.anchor_train_tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
