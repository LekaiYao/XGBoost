import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uproot

from utils.selection import apply_selection


def require_hep_ml():
    try:
        from hep_ml.reweight import FoldingReweighter, GBReweighter
    except ImportError as exc:
        raise RuntimeError(
            "hep_ml is required for reweighting. Install requirements.txt in the project virtualenv."
        ) from exc
    return FoldingReweighter, GBReweighter


def require_hep_ml_classifier():
    try:
        from hep_ml.gradientboosting import UGradientBoostingClassifier
        from hep_ml.losses import LogLossFunction
    except ImportError as exc:
        raise RuntimeError(
            "hep_ml is required for signed-weight domain classification."
        ) from exc
    return UGradientBoostingClassifier, LogLossFunction


def load_tree_frame(path, tree):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ROOT input: {path}")
    with uproot.open(path) as root_file:
        if tree not in root_file:
            raise KeyError(f"Missing TTree '{tree}' in {path}")
        return root_file[tree].arrays(library="pd")


def select_frame(frame, selection, label):
    selected = apply_selection(frame, selection, label)
    if selected.empty:
        raise ValueError(f"{label} removed all events")
    return selected


def validate_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [column for column in columns if not np.isfinite(frame[column].to_numpy(dtype=float)).all()]
        raise ValueError(f"{label} contains non-finite values in: {bad}")


def resolve_weights(frame, branch, label):
    if branch is None:
        return np.ones(len(frame), dtype=float)
    if branch not in frame.columns:
        raise ValueError(f"{label} is missing weight branch '{branch}'")
    weights = frame[branch].to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError(f"{label} weight branch '{branch}' contains non-finite values")
    if float(weights.sum()) <= 0.0:
        raise ValueError(f"{label} weights must have a positive sum")
    return weights


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.square(weights).sum())
    if denominator <= 0.0:
        return 0.0
    return float(weights.sum() ** 2 / denominator)


def weight_summary(weights):
    weights = np.asarray(weights, dtype=float)
    return {
        "entries": int(len(weights)),
        "sum": float(weights.sum()),
        "sum_squares": float(np.square(weights).sum()),
        "effective_sample_size": effective_sample_size(weights),
        "negative_entries": int(np.count_nonzero(weights < 0.0)),
        "negative_fraction": float(np.mean(weights < 0.0)),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
    }


def weighted_cdf_distance(original, target, original_weight, target_weight):
    original = np.asarray(original, dtype=float)
    target = np.asarray(target, dtype=float)
    original_weight = np.asarray(original_weight, dtype=float)
    target_weight = np.asarray(target_weight, dtype=float)
    if original_weight.sum() <= 0.0 or target_weight.sum() <= 0.0:
        raise ValueError("CDF distance requires positive total weights")
    points = np.unique(np.concatenate([original, target]))
    original_order = np.argsort(original)
    target_order = np.argsort(target)
    original_cdf = np.cumsum(original_weight[original_order]) / original_weight.sum()
    target_cdf = np.cumsum(target_weight[target_order]) / target_weight.sum()
    original_at_points = np.zeros(len(points), dtype=float)
    target_at_points = np.zeros(len(points), dtype=float)
    original_indices = np.searchsorted(original[original_order], points, side="right") - 1
    target_indices = np.searchsorted(target[target_order], points, side="right") - 1
    valid_original = original_indices >= 0
    valid_target = target_indices >= 0
    original_at_points[valid_original] = original_cdf[original_indices[valid_original]]
    target_at_points[valid_target] = target_cdf[target_indices[valid_target]]
    return float(np.max(np.abs(original_at_points - target_at_points)))


def build_diagnostics(original, target, variables, original_before, original_after, target_weight):
    per_variable = {}
    for variable in variables:
        before = weighted_cdf_distance(
            original[variable],
            target[variable],
            original_before,
            target_weight,
        )
        after = weighted_cdf_distance(
            original[variable],
            target[variable],
            original_after,
            target_weight,
        )
        per_variable[variable] = {
            "cdf_distance_before": before,
            "cdf_distance_after": after,
            "improvement": before - after,
        }
    return {
        "metric_note": (
            "CDF distances with signed target weights are descriptive effect sizes, not KS p-values."
        ),
        "weights": {
            "original_before": weight_summary(original_before),
            "original_after": weight_summary(original_after),
            "target": weight_summary(target_weight),
        },
        "variables": per_variable,
    }


def train_folding_reweighter(
    original,
    target,
    variables,
    original_weight,
    target_weight,
    *,
    n_folds=5,
    random_state=42,
    n_estimators=40,
    learning_rate=0.2,
    max_depth=3,
    min_samples_leaf=200,
    loss_regularization=5.0,
    subsample=0.8,
):
    FoldingReweighter, GBReweighter = require_hep_ml()
    base = GBReweighter(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        loss_regularization=loss_regularization,
        gb_args={"subsample": subsample, "random_state": random_state},
    )
    model = FoldingReweighter(
        base,
        n_folds=n_folds,
        random_state=random_state,
        verbose=False,
    )
    original_values = original[variables].to_numpy(dtype=float)
    target_values = target[variables].to_numpy(dtype=float)
    model.fit(
        original=original_values,
        target=target_values,
        original_weight=original_weight,
        target_weight=target_weight,
    )
    out_of_fold_weight = model.predict_weights(
        original_values,
        original_weight=original_weight,
        vote_function=lambda predictions: np.mean(predictions, axis=0),
    )
    return model, np.asarray(out_of_fold_weight, dtype=float)


def predict_reweight(model, frame, variables, original_weight=None):
    values = frame[variables].to_numpy(dtype=float)
    kwargs = {"vote_function": lambda predictions: np.mean(predictions, axis=0)}
    if original_weight is not None:
        kwargs["original_weight"] = np.asarray(original_weight, dtype=float)
    return np.asarray(model.predict_weights(values, **kwargs), dtype=float)


def three_way_split_indices(size, random_state=42, reweighter_fraction=0.5):
    if size < 3:
        raise ValueError("Three-way holdout validation requires at least three events")
    if not 0.0 < reweighter_fraction < 1.0:
        raise ValueError("reweighter_fraction must be between zero and one")
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(size)
    reweighter_end = int(round(size * reweighter_fraction))
    reweighter_end = min(max(reweighter_end, 1), size - 2)
    remaining = size - reweighter_end
    domain_train_end = reweighter_end + remaining // 2
    if domain_train_end == reweighter_end:
        domain_train_end += 1
    return {
        "reweighter_train": indices[:reweighter_end],
        "domain_train": indices[reweighter_end:domain_train_end],
        "domain_test": indices[domain_train_end:],
    }


def signed_weighted_auc(mc_score, target_score, mc_weight, target_weight):
    """Rank AUC for target=1, allowing signed target weights."""
    mc_score = np.asarray(mc_score, dtype=float)
    target_score = np.asarray(target_score, dtype=float)
    mc_weight = np.asarray(mc_weight, dtype=float)
    target_weight = np.asarray(target_weight, dtype=float)
    mc_total = float(mc_weight.sum())
    target_total = float(target_weight.sum())
    if mc_total <= 0.0 or target_total <= 0.0:
        raise ValueError("AUC requires positive total weight in both domains")

    order = np.argsort(mc_score, kind="mergesort")
    sorted_score = mc_score[order]
    sorted_weight = mc_weight[order]
    cumulative = np.concatenate([[0.0], np.cumsum(sorted_weight)])
    left = np.searchsorted(sorted_score, target_score, side="left")
    right = np.searchsorted(sorted_score, target_score, side="right")
    lower_weight = cumulative[left]
    tie_weight = cumulative[right] - cumulative[left]
    return float(
        np.sum(target_weight * (lower_weight + 0.5 * tie_weight))
        / (mc_total * target_total)
    )


def train_domain_classifier(
    mc,
    target,
    variables,
    mc_weight,
    target_weight,
    *,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=200,
    subsample=0.8,
):
    UGradientBoostingClassifier, LogLossFunction = require_hep_ml_classifier()
    features = np.concatenate(
        [mc[variables].to_numpy(dtype=float), target[variables].to_numpy(dtype=float)]
    )
    labels = np.concatenate(
        [np.zeros(len(mc), dtype=int), np.ones(len(target), dtype=int)]
    )
    weights = np.concatenate(
        [np.asarray(mc_weight, dtype=float), np.asarray(target_weight, dtype=float)]
    )
    classifier = UGradientBoostingClassifier(
        loss=LogLossFunction(),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
    )
    classifier.fit(features, labels, sample_weight=weights)
    return classifier


def evaluate_domain_classifier(classifier, mc, target, variables, mc_weight, target_weight):
    mc_score = classifier.predict_proba(mc[variables].to_numpy(dtype=float))[:, 1]
    target_score = classifier.predict_proba(target[variables].to_numpy(dtype=float))[:, 1]
    auc = signed_weighted_auc(mc_score, target_score, mc_weight, target_weight)
    return {
        "signed_auc": auc,
        "distance_from_random": abs(auc - 0.5),
        "mc_score_mean": float(np.average(mc_score, weights=mc_weight)),
        "target_score_mean_signed": float(np.average(target_score, weights=target_weight)),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
