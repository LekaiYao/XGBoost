import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uproot

from utils.paths import (
    reweighted_root_path,
    reweighter_model_path,
    reweighting_manifest_path,
)
from utils.selection import apply_selection
from workflows.reweighting.core import predict_reweight, validate_columns, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a trained reweighter to a ROOT TTree.")
    parser.add_argument("reweight_tag")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-tree", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--output-tree")
    parser.add_argument("--selection")
    parser.add_argument("--input-weight-branch")
    parser.add_argument("--output-weight-branch", default="Reweight")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(reweighter_model_path(args.reweight_tag))
    manifest_path = Path(reweighting_manifest_path(args.reweight_tag))
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing trained reweighter artifacts for '{args.reweight_tag}'")
    manifest = json.loads(manifest_path.read_text())
    variables = manifest["variables"]
    selection = args.selection if args.selection is not None else manifest.get("selection")

    input_path = Path(args.input_root)
    with uproot.open(input_path) as root_file:
        if args.input_tree not in root_file:
            raise KeyError(f"Missing TTree '{args.input_tree}' in {input_path}")
        arrays = root_file[args.input_tree].arrays(library="np")
    frame = pd.DataFrame(arrays)
    validate_columns(frame, variables, "apply sample")
    selected = apply_selection(frame, selection, "apply selection")
    if selected.empty:
        raise ValueError("apply selection removed all events")

    initial_weight = None
    full_initial_weight = np.ones(len(frame), dtype=np.float64)
    if args.input_weight_branch is not None:
        if args.input_weight_branch not in frame.columns:
            raise ValueError(f"Missing input weight branch '{args.input_weight_branch}'")
        full_initial_weight = frame[args.input_weight_branch].to_numpy(dtype=float)
        if not np.isfinite(full_initial_weight).all():
            raise ValueError(f"Input weight branch '{args.input_weight_branch}' contains non-finite values")
        initial_weight = selected[args.input_weight_branch].to_numpy(dtype=float)
    model = joblib.load(model_path)
    corrected = predict_reweight(model, selected, variables, initial_weight)
    output_weight = full_initial_weight.copy()
    output_weight[selected.index.to_numpy()] = corrected

    if args.output_weight_branch in arrays:
        raise ValueError(f"Output branch '{args.output_weight_branch}' already exists")
    arrays[args.output_weight_branch] = output_weight
    output_path = Path(args.output_root or reweighted_root_path(args.reweight_tag, input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tree = args.output_tree or args.input_tree
    with uproot.recreate(output_path) as root_file:
        root_file.mktree(output_tree, arrays)

    apply_manifest = {
        "schema_version": 1,
        "reweight_tag": args.reweight_tag,
        "input": {
            "path": str(input_path.resolve()),
            "tree": args.input_tree,
            "entries": len(frame),
        },
        "selection": selection,
        "variables": variables,
        "output": {
            "path": str(output_path.resolve()),
            "tree": output_tree,
            "weight_branch": args.output_weight_branch,
            "weighted_entries": len(selected),
            "unity_entries": len(frame) - len(selected),
        },
    }
    write_json(output_path.with_suffix(".manifest.json"), apply_manifest)
    print(f"Reweighted ROOT: {output_path}")


if __name__ == "__main__":
    main()
