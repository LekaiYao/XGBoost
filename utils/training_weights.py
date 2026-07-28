import numpy as np


def resolve_training_weights(frame, branch, label):
    if branch is None:
        return np.ones(len(frame), dtype=float)
    if branch not in frame.columns:
        raise ValueError(f"{label} is missing training weight branch '{branch}'")
    weights = frame[branch].to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError(f"{label} training weight branch '{branch}' contains non-finite values")
    if np.any(weights < 0.0):
        raise ValueError(f"{label} training weights must be non-negative for XGBoost")
    if float(weights.sum()) <= 0.0:
        raise ValueError(f"{label} training weights must have a positive sum")
    return weights


def balanced_scale_pos_weight(labels, weights):
    labels = np.asarray(labels, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    signal_sum = float(weights[labels].sum())
    background_sum = float(weights[~labels].sum())
    if signal_sum <= 0.0 or background_sum <= 0.0:
        raise ValueError("Both signal and background require positive total training weight")
    return background_sum / signal_sum


def weighted_ks_curve(y_true, score, weights, thresholds=None):
    labels = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 500)

    def cdf(mask):
        ordered = np.argsort(score[mask])
        values = score[mask][ordered]
        current_weights = weights[mask][ordered]
        total = float(current_weights.sum())
        if total <= 0.0:
            raise ValueError("KS curve requires positive total class weight")
        cumulative = np.cumsum(current_weights) / total
        indices = np.searchsorted(values, thresholds, side="right") - 1
        result = np.zeros(len(thresholds), dtype=float)
        valid = indices >= 0
        result[valid] = cumulative[indices[valid]]
        return result

    signal_cdf = cdf(labels)
    background_cdf = cdf(~labels)
    return {
        "score_thresholds": [float(value) for value in thresholds],
        "sig_cdf": [float(value) for value in signal_cdf],
        "bkg_cdf": [float(value) for value in background_cdf],
        "ks_stat": float(np.max(np.abs(signal_cdf - background_cdf))),
    }
