import numpy as np


def resolve_training_weights(frame, branch, label):
    if isinstance(branch, (list, tuple)):
        branches = list(branch)
        if len(set(branches)) != len(branches):
            raise ValueError("Duplicate training weight factors would double-count weights")
        weights = np.ones(len(frame), dtype=float)
        for factor in branches:
            weights *= resolve_training_weights(frame, factor, label)
        if not np.isfinite(weights).all() or not np.isfinite(weights.sum()) or weights.sum() <= 0:
            raise ValueError(f"{label} combined training weights require a finite positive sum")
        return weights
    if branch is None:
        return np.ones(len(frame), dtype=float)
    if branch not in frame.columns:
        raise ValueError(f"{label} is missing training weight branch '{branch}'")
    weights = frame[branch].to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError(f"{label} training weight branch '{branch}' contains non-finite values")
    if np.any(weights < 0.0):
        raise ValueError(f"{label} training weights must be non-negative for XGBoost")
    if not np.isfinite(weights.sum()) or float(weights.sum()) <= 0.0:
        raise ValueError(f"{label} training weights must have a positive sum")
    return weights


def training_weight_branches(base_branch, correction_branch=None):
    """Independent generator and data/MC correction factors, each used once."""
    branches = [branch for branch in (base_branch, correction_branch) if branch]
    if len(set(branches)) != len(branches):
        raise ValueError("Duplicate training weight factors")
    return branches


def weight_branches_from_metadata(config, prefix="signal"):
    """New explicit factor list takes precedence; old model metadata stays valid."""
    key = f"{prefix}_weight_branches"
    if key in config:
        return list(config[key])
    branch = config.get(f"{prefix}_weight_branch")
    return [branch] if branch else []


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
