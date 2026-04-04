import json
import os
from datetime import datetime

from paths import (
    feature_importance_cumulative_path,
    feature_importance_path,
    model_config_path,
    model_dir,
    model_path,
    scaler_path,
    training_score_path,
)


def metadata_path(train_tag):
    return os.path.join(model_dir(train_tag), "run_metadata.json")


def infer_training_mode(train_tag):
    return os.environ.get("TRAINING_MODE") or ("condor" if train_tag.startswith("cnd_") else "local")


def infer_tag_convention_version(train_tag):
    return "legacy" if train_tag.startswith("cnd_") else "v1"


def infer_sample_tag(train_tag):
    parts = train_tag.split("_")
    if train_tag.startswith("cnd_") and len(parts) > 1:
        return parts[1]
    return parts[0]


def infer_feature_set_tag(train_tag, feature_count):
    for part in train_tag.split("_"):
        if part.endswith("var") or part.endswith("v"):
            return part
    return f"{feature_count}var"


def infer_dataset_label(*paths):
    joined = " ".join(paths)
    if "PbPb23" in joined:
        return "PbPb23"
    if "ppRef24" in joined:
        return "ppRef24"
    return "unknown"


def build_artifacts(train_tag):
    return {
        "model_path": model_path(train_tag),
        "scaler_path": scaler_path(train_tag),
        "model_config_path": model_config_path(train_tag),
        "score_plot_path": training_score_path(train_tag),
        "feature_importance_path": feature_importance_path(train_tag),
        "feature_importance_cumulative_path": feature_importance_cumulative_path(train_tag),
    }


def _normalize_mapping(mapping):
    normalized = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            normalized[key] = _normalize_mapping(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                float(item) if isinstance(item, float) else item
                for item in value
            ]
        elif isinstance(value, float):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def save_run_metadata(
    train_tag,
    training_script,
    signal_path,
    background_path,
    signal_selection,
    background_selection,
    input_columns,
    trans_columns,
    pos_weight,
    fixed_model_params,
    best_model_params,
    is_optuna=False,
    optuna_n_trials=None,
    optimized_hyperparameters=None,
    hyperparameter_search_space=None,
    optimization_metric=None,
    best_objective_value=None,
    notes=None,
):
    if optimized_hyperparameters is None:
        optimized_hyperparameters = []
    if hyperparameter_search_space is None:
        hyperparameter_search_space = {}
    if notes is None:
        notes = {}

    metadata = {
        "train_tag": train_tag,
        "tag_convention_version": infer_tag_convention_version(train_tag),
        "training_mode": infer_training_mode(train_tag),
        "training_script": training_script,
        "created_at": datetime.now().isoformat(),
        "is_optuna": is_optuna,
        "optuna_n_trials": optuna_n_trials,
        "optimized_hyperparameters": optimized_hyperparameters,
        "hyperparameter_search_space": _normalize_mapping(hyperparameter_search_space),
        "fixed_model_params": _normalize_mapping(fixed_model_params),
        "sample_tag": infer_sample_tag(train_tag),
        "dataset_label": infer_dataset_label(signal_path, background_path),
        "signal_path": signal_path,
        "background_path": background_path,
        "signal_selection": signal_selection,
        "background_selection": background_selection,
        "feature_set_tag": infer_feature_set_tag(train_tag, len(input_columns)),
        "input_columns": input_columns,
        "trans_columns": trans_columns,
        "feature_count": len(input_columns),
        "train_fraction": 0.8,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "class_weighting": {
            "method": "scale_pos_weight",
            "value": float(pos_weight),
        },
        "best_model_params": _normalize_mapping(best_model_params),
        "optimization_metric": optimization_metric,
        "best_objective_value": float(best_objective_value) if best_objective_value is not None else None,
        "artifacts": build_artifacts(train_tag),
        "notes": {
            "train_tag_note": "legacy tag before new naming convention" if train_tag.startswith("cnd_") else "tag follows current naming convention",
            "naming_recommendation": "<sample>_<nvar>v_<mode><n>_<version>",
            **notes,
        },
    }

    output_path = metadata_path(train_tag)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata saved to: {output_path}")
