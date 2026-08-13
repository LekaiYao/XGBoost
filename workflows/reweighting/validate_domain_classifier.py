import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.paths import ensure_dir, reweighting_dir, reweighting_domain_closure_path
from utils.varsets import get_reweight_varset_columns
from utils.selection import selection_columns
from workflows.reweighting.core import (
    evaluate_domain_classifier,
    load_tree_frame,
    predict_reweight,
    resolve_weights,
    select_frame,
    three_way_split_indices,
    train_domain_classifier,
    train_folding_reweighter,
    validate_columns,
    weight_summary,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent three-way holdout domain-classifier closure test."
    )
    parser.add_argument("reweight_tag")
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--original-tree", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--target-tree", required=True)
    parser.add_argument("--original-weight-branch")
    parser.add_argument("--target-weight-branch", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--variable-set", default="R5")
    parser.add_argument("--classifier-variable-set", default="R8")
    parser.add_argument("--sample", default="pp")
    parser.add_argument("--channel", default="X")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--reweighter-fraction", type=float, default=0.5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--rw-n-estimators", type=int, default=40)
    parser.add_argument("--classifier-n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--classifier-learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--loss-regularization", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    return parser.parse_args()


def take(frame, weights, indices):
    return frame.iloc[indices].reset_index(drop=True), np.asarray(weights)[indices]


def plot_domain_scores(
    before_classifier,
    after_classifier,
    mc,
    target,
    variables,
    mc_weight_before,
    mc_weight_after,
    target_weight,
    output,
):
    mc_values = mc[variables].to_numpy(dtype=float)
    target_values = target[variables].to_numpy(dtype=float)
    classifiers = [("Before reweighting", before_classifier), ("After R5", after_classifier)]
    mc_weights = [mc_weight_before, mc_weight_after]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    bins = np.linspace(0.0, 1.0, 31)
    for axis, (title, classifier), current_mc_weight in zip(
        axes, classifiers, mc_weights
    ):
        mc_score = classifier.predict_proba(mc_values)[:, 1]
        target_score = classifier.predict_proba(target_values)[:, 1]
        axis.hist(
            mc_score,
            bins=bins,
            weights=current_mc_weight / np.sum(current_mc_weight),
            histtype="step",
            linewidth=1.6,
            color="#4477AA",
            label="MC",
        )
        axis.hist(
            target_score,
            bins=bins,
            weights=target_weight / np.sum(target_weight),
            histtype="step",
            linewidth=1.6,
            color="black",
            label="DATA sWeight",
        )
        axis.set_title(title)
        axis.set_xlabel("Domain-classifier score")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Normalized entries")
    figure.suptitle("Independent R8 domain-classifier holdout")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main():
    args = parse_args()
    rw_variables = get_reweight_varset_columns(args.sample, args.variable_set, args.channel)
    classifier_variables = get_reweight_varset_columns(
        args.sample, args.classifier_variable_set, args.channel
    )
    all_variables = list(dict.fromkeys([*rw_variables, *classifier_variables]))
    required_columns = list(
        dict.fromkeys([*all_variables, *selection_columns(args.selection)])
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
    validate_columns(original, all_variables, "original sample")
    validate_columns(target, all_variables, "target sample")
    original_weight = resolve_weights(original, args.original_weight_branch, "original sample")
    target_weight = resolve_weights(target, args.target_weight_branch, "target sample")

    original_indices = three_way_split_indices(
        len(original), args.random_state, args.reweighter_fraction
    )
    target_indices = three_way_split_indices(
        len(target), args.random_state + 1, args.reweighter_fraction
    )
    original_parts = {
        name: take(original, original_weight, indices)
        for name, indices in original_indices.items()
    }
    target_parts = {
        name: take(target, target_weight, indices)
        for name, indices in target_indices.items()
    }

    rw_original, rw_original_weight = original_parts["reweighter_train"]
    rw_target, rw_target_weight = target_parts["reweighter_train"]
    reweighter, _ = train_folding_reweighter(
        rw_original,
        rw_target,
        rw_variables,
        rw_original_weight,
        rw_target_weight,
        n_folds=args.n_folds,
        random_state=args.random_state,
        n_estimators=args.rw_n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        loss_regularization=args.loss_regularization,
        subsample=args.subsample,
    )

    corrected_weights = {}
    for part in ("domain_train", "domain_test"):
        frame, base_weight = original_parts[part]
        corrected_weights[part] = predict_reweight(
            reweighter, frame, rw_variables, base_weight
        )

    mc_train, mc_train_weight = original_parts["domain_train"]
    target_train, target_train_weight = target_parts["domain_train"]
    classifier_parameters = {
        "random_state": args.random_state + 10,
        "n_estimators": args.classifier_n_estimators,
        "learning_rate": args.classifier_learning_rate,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "subsample": args.subsample,
    }
    before_classifier = train_domain_classifier(
        mc_train,
        target_train,
        classifier_variables,
        mc_train_weight,
        target_train_weight,
        **classifier_parameters,
    )
    after_classifier = train_domain_classifier(
        mc_train,
        target_train,
        classifier_variables,
        corrected_weights["domain_train"],
        target_train_weight,
        **classifier_parameters,
    )

    mc_test, mc_test_weight = original_parts["domain_test"]
    target_test, target_test_weight = target_parts["domain_test"]
    before = evaluate_domain_classifier(
        before_classifier,
        mc_test,
        target_test,
        classifier_variables,
        mc_test_weight,
        target_test_weight,
    )
    after = evaluate_domain_classifier(
        after_classifier,
        mc_test,
        target_test,
        classifier_variables,
        corrected_weights["domain_test"],
        target_test_weight,
    )
    domain_plot = (
        Path(reweighting_dir(args.reweight_tag))
        / f"domain_classifier_scores_seed{args.random_state}.pdf"
    )
    plot_domain_scores(
        before_classifier,
        after_classifier,
        mc_test,
        target_test,
        classifier_variables,
        mc_test_weight,
        corrected_weights["domain_test"],
        target_test_weight,
        domain_plot,
    )
    payload = {
        "schema_version": 1,
        "reweight_tag": args.reweight_tag,
        "method": "three_way_holdout_signed_weight_domain_classifier",
        "metric_note": (
            "Target sWeights remain signed. signed_auc is a weighted rank diagnostic; "
            "0.5 is random-domain closure and it is not a conventional probability AUC."
        ),
        "split": {
            "reweighter_train_fraction": args.reweighter_fraction,
            "domain_train_fraction": (1.0 - args.reweighter_fraction) / 2.0,
            "domain_test_fraction": (1.0 - args.reweighter_fraction) / 2.0,
            "random_state": args.random_state,
            "entries": {
                name: {
                    "original": len(original_parts[name][0]),
                    "target": len(target_parts[name][0]),
                }
                for name in original_parts
            },
        },
        "reweighter": {
            "variable_set": args.variable_set,
            "variables": rw_variables,
        },
        "domain_classifier": {
            "variable_set": args.classifier_variable_set,
            "variables": classifier_variables,
            "parameters": classifier_parameters,
        },
        "holdout": {
            "before": before,
            "after": after,
            "distance_reduction": before["distance_from_random"]
            - after["distance_from_random"],
            "mc_weight_after": weight_summary(corrected_weights["domain_test"]),
            "target_weight": weight_summary(target_test_weight),
        },
        "inputs": {
            "original": {
                "path": str(Path(args.original_root).resolve()),
                "tree": args.original_tree,
                "weight_branch": args.original_weight_branch,
            },
            "target": {
                "path": str(Path(args.target_root).resolve()),
                "tree": args.target_tree,
                "weight_branch": args.target_weight_branch,
            },
            "selection": args.selection,
        },
    }
    ensure_dir(reweighting_dir(args.reweight_tag))
    output = write_json(reweighting_domain_closure_path(args.reweight_tag), payload)
    print(f"Before signed holdout AUC: {before['signed_auc']:.6f}")
    print(f"After signed holdout AUC:  {after['signed_auc']:.6f}")
    print(f"Domain score plot: {domain_plot}")
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
