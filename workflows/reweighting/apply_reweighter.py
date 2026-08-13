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
from utils.selection import apply_selection, selection_columns
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
    parser.add_argument("--output-columns", help="Comma-separated slim output branches; selection/model branches are added automatically")
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
    requested_columns = [x.strip() for x in args.output_columns.split(",")] if args.output_columns else None
    read_columns = None
    if requested_columns is not None:
        read_columns = list(dict.fromkeys(requested_columns + variables + selection_columns(selection) + ([args.input_weight_branch] if args.input_weight_branch else [])))
    model = joblib.load(model_path)
    if requested_columns is not None:
        output_path = Path(args.output_root or reweighted_root_path(args.reweight_tag, input_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_tree = args.output_tree or args.input_tree
        total_entries = selected_entries = 0
        with uproot.recreate(output_path) as output_file:
            output_handle = None
            for arrays in uproot.iterate(f"{input_path}:{args.input_tree}", expressions=read_columns, library="np", step_size=1000):
                frame = pd.DataFrame(arrays)
                validate_columns(frame, variables, "apply sample")
                selected = apply_selection(frame, selection, "apply selection")
                initial = selected[args.input_weight_branch].to_numpy(dtype=float) if args.input_weight_branch else None
                output_weight = frame[args.input_weight_branch].to_numpy(dtype=float).copy() if args.input_weight_branch else np.ones(len(frame), dtype=np.float64)
                if len(selected):
                    output_weight[selected.index.to_numpy()] = predict_reweight(model, selected, variables, initial)
                chunk = {name: arrays[name] for name in read_columns}
                chunk[args.output_weight_branch] = output_weight
                if output_handle is None:
                    output_file.mktree(output_tree, chunk)
                    output_handle = output_file[output_tree]
                else:
                    output_handle.extend(chunk)
                total_entries += len(frame); selected_entries += len(selected)
        apply_manifest = {
            "schema_version": 1, "reweight_tag": args.reweight_tag,
            "input": {"path": str(input_path.resolve()), "tree": args.input_tree, "entries": total_entries},
            "selection": selection, "variables": variables,
            "output": {"path": str(output_path.resolve()), "tree": output_tree, "weight_branch": args.output_weight_branch,
                       "weighted_entries": selected_entries, "unity_entries": total_entries - selected_entries,
                       "columns": list(dict.fromkeys(read_columns + [args.output_weight_branch]))},
        }
        write_json(output_path.with_suffix(".manifest.json"), apply_manifest)
        print(f"Reweighted ROOT: {output_path}")
        return
    with uproot.open(input_path) as root_file:
        if args.input_tree not in root_file:
            raise KeyError(f"Missing TTree '{args.input_tree}' in {input_path}")
        arrays = root_file[args.input_tree].arrays(read_columns, library="np")
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
