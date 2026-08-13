import numpy as np


def weighted_efficiency_thresholds(scores, weights, targets):
    """Return strict score thresholds and their achieved weighted efficiencies."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    targets = [float(target) for target in targets]
    if scores.ndim != 1 or weights.ndim != 1 or len(scores) != len(weights):
        raise ValueError("scores and weights must be one-dimensional with equal length")
    if len(scores) == 0:
        raise ValueError("Cannot determine thresholds from an empty reference sample")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError("Reference scores and weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("Reference weights must be strictly positive")
    if any(not 0.0 < target < 1.0 for target in targets):
        raise ValueError("Efficiency targets must lie strictly between zero and one")

    total_weight = float(weights.sum())
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    cumulative = np.cumsum(weights[order]) / total_weight
    rows = []
    for target in targets:
        index = int(np.searchsorted(cumulative, 1.0 - target, side="left"))
        index = min(index, len(sorted_scores) - 1)
        threshold = float(sorted_scores[index])
        achieved = float(weights[scores > threshold].sum() / total_weight)
        rows.append(
            {
                "target_efficiency": target,
                "score_threshold": threshold,
                "achieved_efficiency": achieved,
            }
        )
    return rows
