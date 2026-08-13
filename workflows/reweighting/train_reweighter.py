import argparse
from pathlib import Path

from utils.paths import (
    ensure_dir,
    reweighter_model_path,
    reweighting_diagnostics_path,
    reweighting_dir,
    reweighting_manifest_path,
)
from utils.varsets import get_reweight_varset_columns
from utils.selection import selection_columns
from workflows.reweighting.core import (
    build_diagnostics,
    load_tree_frame,
    resolve_weights,
    save_model,
    select_frame,
    train_folding_reweighter,
    validate_columns,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a folding GBReweighter from ROOT TTrees.")
    parser.add_argument("reweight_tag")
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--original-tree", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--target-tree", required=True)
    parser.add_argument("--original-weight-branch")
    parser.add_argument("--target-weight-branch", required=True)
    parser.add_argument("--selection")
    variable_group = parser.add_mutually_exclusive_group(required=True)
    variable_group.add_argument("--variable-set")
    variable_group.add_argument("--variables", nargs="+")
    parser.add_argument("--validation-variable-set", default="R8")
    parser.add_argument("--sample", default="pp")
    parser.add_argument("--channel", default="X")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--loss-regularization", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--physics-status", default="technical_prototype")
    return parser.parse_args()


def resolve_variables(args):
    if args.variables:
        variables = list(dict.fromkeys(args.variables))
        variable_set = "custom"
    else:
        variables = get_reweight_varset_columns(args.sample, args.variable_set, args.channel)
        variable_set = args.variable_set
    validation_variables = get_reweight_varset_columns(
        args.sample,
        args.validation_variable_set,
        args.channel,
    )
    validation_variables = list(dict.fromkeys([*validation_variables, *variables]))
    return variable_set, variables, validation_variables


def main():
    args = parse_args()
    variable_set, variables, validation_variables = resolve_variables(args)
    required_columns = list(
        dict.fromkeys([*validation_variables, *selection_columns(args.selection)])
    )
    original = select_frame(
        load_tree_frame(
            args.original_root,
            args.original_tree,
            [*required_columns, *([args.original_weight_branch] if args.original_weight_branch else [])],
        ),
        args.selection,
        "original selection",
    )
    target = select_frame(
        load_tree_frame(
            args.target_root,
            args.target_tree,
            [*required_columns, args.target_weight_branch],
        ),
        args.selection,
        "target selection",
    )
    all_columns = list(dict.fromkeys([*variables, *validation_variables]))
    validate_columns(original, all_columns, "original sample")
    validate_columns(target, all_columns, "target sample")
    original_weight = resolve_weights(original, args.original_weight_branch, "original sample")
    target_weight = resolve_weights(target, args.target_weight_branch, "target sample")

    model, corrected_weight = train_folding_reweighter(
        original,
        target,
        variables,
        original_weight,
        target_weight,
        n_folds=args.n_folds,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        loss_regularization=args.loss_regularization,
        subsample=args.subsample,
    )
    diagnostics = build_diagnostics(
        original,
        target,
        validation_variables,
        original_weight,
        corrected_weight,
        target_weight,
    )

    ensure_dir(reweighting_dir(args.reweight_tag))
    model_path = save_model(reweighter_model_path(args.reweight_tag), model)
    diagnostics_path = write_json(
        reweighting_diagnostics_path(args.reweight_tag),
        diagnostics,
    )
    parameters = {
        "n_folds": args.n_folds,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "loss_regularization": args.loss_regularization,
        "subsample": args.subsample,
    }
    manifest = {
        "schema_version": 1,
        "reweight_tag": args.reweight_tag,
        "physics_status": args.physics_status,
        "algorithm": "hep_ml.FoldingReweighter(GBReweighter)",
        "variable_set": variable_set,
        "variables": variables,
        "validation_variables": validation_variables,
        "selection": args.selection,
        "inputs": {
            "original": {
                "path": str(Path(args.original_root).resolve()),
                "tree": args.original_tree,
                "weight_branch": args.original_weight_branch,
                "entries_after_selection": len(original),
            },
            "target": {
                "path": str(Path(args.target_root).resolve()),
                "tree": args.target_tree,
                "weight_branch": args.target_weight_branch,
                "entries_after_selection": len(target),
            },
        },
        "parameters": parameters,
        "artifacts": {
            "model": Path(model_path).name,
            "diagnostics": Path(diagnostics_path).name,
        },
    }
    manifest_path = write_json(reweighting_manifest_path(args.reweight_tag), manifest)
    print(f"Reweighter trained: {args.reweight_tag}")
    print(f"Model: {model_path}")
    print(f"Diagnostics: {diagnostics_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
