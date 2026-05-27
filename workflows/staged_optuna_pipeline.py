import argparse
import json
import os
import re
import time
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import uproot
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_sample_from_tag as infer_sample_from_config,
    infer_selection_profile,
    resolve_training_config,
    to_root_spec,
)
from utils.paths import (
    condor_feature_importance_cumulative_path,
    condor_feature_importance_path,
    condor_model_config_path,
    condor_model_dir,
    condor_model_path,
    condor_scaler_path,
    condor_training_dir,
    condor_training_score_path,
    ensure_dir,
)
from utils.run_metadata import save_run_metadata
from utils.selection import apply_selection
from utils.varsets import VARSETS, get_varset_columns, infer_sample_from_tag, infer_varset_from_tag

CORE_PARAMS = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 4,
}


def int_space(low, high):
    return {"type": "int", "low": low, "high": high}


def float_space(low, high, log=False):
    cfg = {"type": "float", "low": low, "high": high}
    if log:
        cfg["log"] = True
    return cfg


STAGE_CONFIGS = {
    "2v1": {
        "baseline": {"max_depth": 2, "min_child_weight": 10, "gamma": 2.0, "subsample": 0.65, "colsample_bytree": 0.65, "learning_rate": 0.05, "n_estimators": 500, "reg_lambda": 8.0, "reg_alpha": 1.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(2, 3), "min_child_weight": int_space(8, 14), "gamma": float_space(1.0, 3.0)},
        "step2": {"subsample": float_space(0.60, 0.75), "colsample_bytree": float_space(0.60, 0.75)},
        "step3": {"reg_lambda": float_space(5.0, 12.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.08), "n_estimators": int_space(300, 800)},
        "step5": {"scale_pos_weight": float_space(0.5, 3.0)},
    },
    "2v2": {
        "baseline": {"max_depth": 3, "min_child_weight": 10, "gamma": 1.5, "subsample": 0.65, "colsample_bytree": 0.70, "learning_rate": 0.05, "n_estimators": 500, "reg_lambda": 6.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(2, 4), "min_child_weight": int_space(7, 14), "gamma": float_space(0.5, 2.5)},
        "step2": {"subsample": float_space(0.60, 0.80), "colsample_bytree": float_space(0.60, 0.80)},
        "step3": {"reg_lambda": float_space(3.0, 10.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.08), "n_estimators": int_space(300, 900)},
        "step5": {"scale_pos_weight": float_space(0.5, 3.0)},
    },
    "2v3": {
        "baseline": {"max_depth": 3, "min_child_weight": 8, "gamma": 1.0, "subsample": 0.70, "colsample_bytree": 0.70, "learning_rate": 0.05, "n_estimators": 500, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(5, 12), "gamma": float_space(0.0, 2.0)},
        "step2": {"subsample": float_space(0.60, 0.85), "colsample_bytree": float_space(0.60, 0.85)},
        "step3": {"reg_lambda": float_space(2.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(300, 1000)},
        "step5": {"scale_pos_weight": float_space(0.5, 4.0)},
    },
    "2v4": {
        "baseline": {"max_depth": 4, "min_child_weight": 8, "gamma": 1.0, "subsample": 0.70, "colsample_bytree": 0.75, "learning_rate": 0.05, "n_estimators": 600, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(5, 10), "gamma": float_space(0.0, 2.0)},
        "step2": {"subsample": float_space(0.65, 0.85), "colsample_bytree": float_space(0.65, 0.85)},
        "step3": {"reg_lambda": float_space(2.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(400, 1000)},
        "step5": {"scale_pos_weight": float_space(0.5, 4.0)},
    },
    "2v5": {
        "baseline": {"max_depth": 4, "min_child_weight": 6, "gamma": 0.5, "subsample": 0.75, "colsample_bytree": 0.75, "learning_rate": 0.05, "n_estimators": 600, "reg_lambda": 4.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(3, 6), "min_child_weight": int_space(4, 10), "gamma": float_space(0.0, 1.5)},
        "step2": {"subsample": float_space(0.65, 0.90), "colsample_bytree": float_space(0.65, 0.90)},
        "step3": {"reg_lambda": float_space(1.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(400, 1200)},
        "step5": {"scale_pos_weight": float_space(0.5, 5.0)},
    },
    "2v6": {
        "baseline": {"max_depth": 4, "min_child_weight": 5, "gamma": 0.5, "subsample": 0.80, "colsample_bytree": 0.80, "learning_rate": 0.05, "n_estimators": 700, "reg_lambda": 3.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(3, 6), "min_child_weight": int_space(3, 8), "gamma": float_space(0.0, 1.5)},
        "step2": {"subsample": float_space(0.70, 0.95), "colsample_bytree": float_space(0.70, 0.95)},
        "step3": {"reg_lambda": float_space(1.0, 6.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.02, 0.10), "n_estimators": int_space(500, 1200)},
        "step5": {"scale_pos_weight": float_space(0.5, 5.0)},
    },
    "2v7": {
        "baseline": {"max_depth": 4, "min_child_weight": 6, "gamma": 0.0, "subsample": 0.70, "colsample_bytree": 0.70, "learning_rate": 0.05, "n_estimators": 600, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(3, 6), "min_child_weight": int_space(4, 10), "gamma": float_space(0.0, 1.0)},
        "step2": {"subsample": float_space(0.60, 0.85), "colsample_bytree": float_space(0.60, 0.85)},
        "step3": {"reg_lambda": float_space(2.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(400, 1000)},
        "step5": {"scale_pos_weight": float_space(0.5, 4.0)},
    },
    "2v8": {
        "baseline": {"max_depth": 3, "min_child_weight": 4, "gamma": 1.0, "subsample": 0.70, "colsample_bytree": 0.70, "learning_rate": 0.05, "n_estimators": 600, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(2, 8), "gamma": float_space(0.0, 2.0)},
        "step2": {"subsample": float_space(0.60, 0.85), "colsample_bytree": float_space(0.60, 0.85)},
        "step3": {"reg_lambda": float_space(2.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(400, 1000)},
        "step5": {"scale_pos_weight": float_space(0.5, 4.0)},
    },
    "2v9": {
        "baseline": {"max_depth": 5, "min_child_weight": 4, "gamma": 0.0, "subsample": 0.80, "colsample_bytree": 0.80, "learning_rate": 0.05, "n_estimators": 800, "reg_lambda": 3.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(4, 7), "min_child_weight": int_space(2, 6), "gamma": float_space(0.0, 1.0)},
        "step2": {"subsample": float_space(0.70, 0.95), "colsample_bytree": float_space(0.70, 0.95)},
        "step3": {"reg_lambda": float_space(1.0, 6.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.02, 0.08), "n_estimators": int_space(600, 1500)},
        "step5": {"scale_pos_weight": float_space(0.5, 5.0)},
    },
    "2v10": {
        "baseline": {"max_depth": 6, "min_child_weight": 3, "gamma": 0.0, "subsample": 0.80, "colsample_bytree": 0.80, "learning_rate": 0.03, "n_estimators": 1000, "reg_lambda": 3.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0},
        "step1": {"max_depth": int_space(5, 8), "min_child_weight": int_space(2, 5), "gamma": float_space(0.0, 1.0)},
        "step2": {"subsample": float_space(0.70, 0.95), "colsample_bytree": float_space(0.70, 0.95)},
        "step3": {"reg_lambda": float_space(1.0, 5.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.02, 0.06), "n_estimators": int_space(800, 1800)},
        "step5": {"scale_pos_weight": float_space(0.5, 5.0)},
    },
    "v21": {
        "baseline": {"max_depth": 3, "min_child_weight": 10, "gamma": 1.5, "subsample": 0.65, "colsample_bytree": 0.65, "learning_rate": 0.05, "n_estimators": 600, "reg_lambda": 8.0, "reg_alpha": 1.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(2, 4), "min_child_weight": int_space(8, 14), "gamma": float_space(0.8, 2.5)},
        "step2": {"subsample": float_space(0.60, 0.75), "colsample_bytree": float_space(0.60, 0.75)},
        "step3": {"reg_lambda": float_space(5.0, 12.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.08), "n_estimators": int_space(400, 900)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v22": {
        "baseline": {"max_depth": 3, "min_child_weight": 8, "gamma": 1.2, "subsample": 0.70, "colsample_bytree": 0.70, "learning_rate": 0.05, "n_estimators": 600, "reg_lambda": 6.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(2, 4), "min_child_weight": int_space(6, 12), "gamma": float_space(0.5, 2.0)},
        "step2": {"subsample": float_space(0.60, 0.80), "colsample_bytree": float_space(0.60, 0.80)},
        "step3": {"reg_lambda": float_space(4.0, 10.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.08), "n_estimators": int_space(400, 900)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v23": {
        "baseline": {"max_depth": 3, "min_child_weight": 8, "gamma": 1.0, "subsample": 0.70, "colsample_bytree": 0.75, "learning_rate": 0.05, "n_estimators": 700, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(5, 10), "gamma": float_space(0.3, 1.8)},
        "step2": {"subsample": float_space(0.65, 0.85), "colsample_bytree": float_space(0.65, 0.85)},
        "step3": {"reg_lambda": float_space(3.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(500, 1000)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v24": {
        "baseline": {"max_depth": 4, "min_child_weight": 8, "gamma": 1.0, "subsample": 0.70, "colsample_bytree": 0.70, "learning_rate": 0.05, "n_estimators": 700, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(5, 10), "gamma": float_space(0.3, 1.8)},
        "step2": {"subsample": float_space(0.65, 0.85), "colsample_bytree": float_space(0.65, 0.85)},
        "step3": {"reg_lambda": float_space(3.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(500, 1000)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v25": {
        "baseline": {"max_depth": 4, "min_child_weight": 6, "gamma": 0.8, "subsample": 0.75, "colsample_bytree": 0.75, "learning_rate": 0.05, "n_estimators": 800, "reg_lambda": 4.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(4, 9), "gamma": float_space(0.2, 1.5)},
        "step2": {"subsample": float_space(0.65, 0.90), "colsample_bytree": float_space(0.65, 0.90)},
        "step3": {"reg_lambda": float_space(2.0, 7.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(500, 1200)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v26": {
        "baseline": {"max_depth": 4, "min_child_weight": 6, "gamma": 0.5, "subsample": 0.80, "colsample_bytree": 0.80, "learning_rate": 0.05, "n_estimators": 800, "reg_lambda": 4.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 6), "min_child_weight": int_space(4, 8), "gamma": float_space(0.0, 1.2)},
        "step2": {"subsample": float_space(0.70, 0.90), "colsample_bytree": float_space(0.70, 0.90)},
        "step3": {"reg_lambda": float_space(2.0, 6.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(600, 1200)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v27": {
        "baseline": {"max_depth": 5, "min_child_weight": 6, "gamma": 0.5, "subsample": 0.75, "colsample_bytree": 0.75, "learning_rate": 0.05, "n_estimators": 900, "reg_lambda": 4.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(4, 6), "min_child_weight": int_space(4, 8), "gamma": float_space(0.0, 1.2)},
        "step2": {"subsample": float_space(0.65, 0.85), "colsample_bytree": float_space(0.65, 0.85)},
        "step3": {"reg_lambda": float_space(2.0, 6.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.08), "n_estimators": int_space(700, 1300)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v28": {
        "baseline": {"max_depth": 4, "min_child_weight": 5, "gamma": 0.3, "subsample": 0.80, "colsample_bytree": 0.75, "learning_rate": 0.04, "n_estimators": 1000, "reg_lambda": 3.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 6), "min_child_weight": int_space(3, 8), "gamma": float_space(0.0, 1.0)},
        "step2": {"subsample": float_space(0.70, 0.90), "colsample_bytree": float_space(0.65, 0.85)},
        "step3": {"reg_lambda": float_space(1.0, 6.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.02, 0.08), "n_estimators": int_space(700, 1500)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v29": {
        "baseline": {"max_depth": 3, "min_child_weight": 6, "gamma": 0.8, "subsample": 0.85, "colsample_bytree": 0.85, "learning_rate": 0.05, "n_estimators": 800, "reg_lambda": 5.0, "reg_alpha": 0.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(4, 9), "gamma": float_space(0.2, 1.5)},
        "step2": {"subsample": float_space(0.75, 0.95), "colsample_bytree": float_space(0.75, 0.95)},
        "step3": {"reg_lambda": float_space(3.0, 8.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.03, 0.10), "n_estimators": int_space(500, 1200)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
    "v30": {
        "baseline": {"max_depth": 4, "min_child_weight": 7, "gamma": 1.0, "subsample": 0.70, "colsample_bytree": 0.80, "learning_rate": 0.04, "n_estimators": 900, "reg_lambda": 6.0, "reg_alpha": 1.0, "scale_pos_weight": "ratio"},
        "step1": {"max_depth": int_space(3, 5), "min_child_weight": int_space(5, 10), "gamma": float_space(0.3, 1.8)},
        "step2": {"subsample": float_space(0.65, 0.85), "colsample_bytree": float_space(0.70, 0.90)},
        "step3": {"reg_lambda": float_space(4.0, 10.0), "reg_alpha": float_space(0.0, 2.0)},
        "step4": {"learning_rate": float_space(0.02, 0.08), "n_estimators": int_space(700, 1400)},
        "step5": {"scale_pos_weight": float_space(0.5, 2.0)},
    },
}


def clamp(value, low, high):
    return max(low, min(high, value))


def suggest_param(trial, name, config):
    if config["type"] == "int":
        return trial.suggest_int(name, config["low"], config["high"])
    if config["type"] == "float":
        return trial.suggest_float(name, config["low"], config["high"], log=config.get("log", False))
    raise ValueError(f"Unsupported parameter type for {name}: {config['type']}")


def infer_feature_set(train_tag):
    sample_key = infer_sample_from_tag(train_tag)
    candidate = infer_varset_from_tag(train_tag, sample=sample_key)
    if candidate is not None:
        return sample_key, candidate
    raise ValueError(f"Unable to infer feature set from train_tag: {train_tag}")


def infer_stage_group(train_tag, explicit_group):
    if explicit_group:
        return explicit_group
    match = re.search(r"_(2v\d+)$", train_tag)
    if not match:
        raise ValueError("Cannot infer stage group from train_tag; provide --stage-group explicitly.")
    return match.group(1)


def normalize_stage_group(stage_group):
    match = re.fullmatch(r"2v(\d+)", stage_group or "")
    if not match:
        return stage_group
    idx = int(match.group(1))
    if idx <= 0:
        return stage_group
    canonical = ((idx - 1) % 10) + 1
    return f"2v{canonical}"


def get_selection_config(train_tag, dataset_year_override=None, selection_profile_override=None):
    sample = infer_sample_from_config(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset_year = dataset_year_override or infer_dataset_year(train_tag, sample)
    selection_profile = selection_profile_override or infer_selection_profile(train_tag, sample)
    cfg = resolve_training_config(sample, channel, dataset_year, selection_profile)
    return {
        "sample": sample,
        "dataset_year": dataset_year,
        "selection_profile": selection_profile,
        "dataset_source": cfg["dataset_source"],
        "channel": channel,
        "signal_path": to_root_spec(cfg["signal"]),
        "background_path": to_root_spec(cfg["background"]),
        "signal_selection": cfg["signal_selection"],
        "background_selection": cfg["background_selection"],
    }


def robust_ensure_dir(path, retries=3, wait_sec=1.0):
    last_exc = None
    for _ in range(retries):
        try:
            created = ensure_dir(path)
            if os.path.isdir(created):
                return created
        except OSError as exc:
            last_exc = exc
        time.sleep(wait_sec)
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"Failed to create directory after retries: {path}")


def save_feature_importance(model, feature_names, train_tag):
    importance_pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    importance_json_path = condor_feature_importance_path(train_tag)
    with open(importance_json_path, "w") as f:
        json.dump(
            [
                {"rank": rank, "feature": name, "importance": float(score)}
                for rank, (name, score) in enumerate(importance_pairs, start=1)
            ],
            f,
            indent=2,
        )

    ordered_names = [name for name, _ in importance_pairs]
    ordered_scores = np.array([float(score) for _, score in importance_pairs])
    cumulative_percent = np.cumsum(ordered_scores) / np.sum(ordered_scores) * 100.0

    cumulative_plot_path = condor_feature_importance_cumulative_path(train_tag)
    plt.figure(figsize=(8, 5))
    plt.plot(ordered_names, cumulative_percent, color="black", linewidth=1, alpha=0.6)
    plt.scatter(ordered_names, cumulative_percent, color="tab:blue", s=45)
    plt.axhline(95.0, color="tab:red", linestyle="--", linewidth=1.5, label="95% cumulative importance")
    plt.ylabel("Cumulative importance (%)")
    plt.xlabel("Features ordered by importance")
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha="right")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cumulative_plot_path)
    plt.close()


def build_step6_space(current_params):
    max_depth = int(round(current_params["max_depth"]))
    min_child = int(round(current_params["min_child_weight"]))
    gamma = float(current_params["gamma"])
    subsample = float(current_params["subsample"])
    colsample = float(current_params["colsample_bytree"])
    return {
        "max_depth": int_space(max(2, max_depth - 1), max_depth + 1),
        "min_child_weight": int_space(max(1, min_child - 2), min_child + 2),
        "gamma": float_space(max(0.0, gamma - 0.5), gamma + 0.5),
        "subsample": float_space(clamp(subsample - 0.05, 0.30, 1.0), clamp(subsample + 0.05, 0.30, 1.0)),
        "colsample_bytree": float_space(clamp(colsample - 0.05, 0.30, 1.0), clamp(colsample + 0.05, 0.30, 1.0)),
    }


def run_stage(step_name, search_space, current_params, n_trials, X_train, y_train, X_val, y_val):
    def objective(trial):
        trial_params = {}
        for param_name, cfg in search_space.items():
            trial_params[param_name] = suggest_param(trial, param_name, cfg)
        full_params = {**CORE_PARAMS, **current_params, **trial_params}
        model = XGBClassifier(**full_params)
        model.fit(X_train, y_train["is_sig"])
        pred = model.predict_proba(X_val)[:, 1]
        val_fpr, val_tpr, _ = roc_curve(y_val["is_sig"].astype(int).to_numpy(), pred)
        val_auc = auc(val_fpr, val_tpr)
        trial.set_user_attr("validation_auc", float(val_auc))
        return val_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CORE_PARAMS["random_state"]),
    )
    study.optimize(objective, n_trials=n_trials)

    updated_params = deepcopy(current_params)
    for key, value in study.best_params.items():
        updated_params[key] = value

    top_trials = sorted(
        [trial for trial in study.trials if trial.value is not None],
        key=lambda trial: trial.value,
        reverse=True,
    )[:5]
    top_payload = [
        {
            "trial_number": trial.number,
            "value": float(trial.value),
            "params": trial.params,
        }
        for trial in top_trials
    ]

    summary = {
        "step": step_name,
        "n_trials": n_trials,
        "search_space": search_space,
        "best_value": float(study.best_value),
        "best_params_scanned": study.best_params,
        "updated_params": updated_params,
        "top_trials": top_payload,
    }
    return updated_params, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_tag")
    parser.add_argument("--stage-group", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-year", choices=["2023", "2024"], default=None)
    args = parser.parse_args()

    train_tag = args.train_tag
    stage_group_requested = infer_stage_group(train_tag, args.stage_group)
    stage_group = normalize_stage_group(stage_group_requested)
    if stage_group not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage group: {stage_group}. Available: {sorted(STAGE_CONFIGS)}")

    n_trials = int(os.environ.get("OPTUNA_N_TRIALS", "200"))
    sample_key, feature_set_tag = infer_feature_set(train_tag)
    channel = infer_channel_from_tag(train_tag)
    input_columns = get_varset_columns(sample_key, feature_set_tag, channel=channel)
    trans_columns = [f"{col}_trans" for col in input_columns]

    robust_ensure_dir(condor_model_dir(train_tag))
    robust_ensure_dir(condor_training_dir(train_tag))
    stages_dir = robust_ensure_dir(os.path.join(condor_training_dir(train_tag), "stages"))
    state_path = os.path.join(stages_dir, "pipeline_state.json")

    print(f"Train tag: {train_tag}")
    print(f"Stage group requested: {stage_group_requested}")
    print(f"Stage group resolved: {stage_group}")
    print(f"Sample: {sample_key}")
    print(f"Feature set: {feature_set_tag}")
    print(f"OPTUNA_N_TRIALS per step: {n_trials}")
    selection_cfg = get_selection_config(train_tag, dataset_year_override=args.dataset_year)
    dataset_source = selection_cfg["dataset_source"]
    sig_path = selection_cfg["signal_path"]
    bkg_path = selection_cfg["background_path"]

    print(f"Dataset source: {dataset_source}")
    print(f"Signal path: {sig_path}")
    print(f"Background path: {bkg_path}")
    signal_selection = selection_cfg["signal_selection"]
    background_selection = selection_cfg["background_selection"]

    print(f"Signal selection: {signal_selection}")
    print(f"Background selection: {background_selection}")

    ak_sig = uproot.concatenate(sig_path, library="pd")
    ak_bkg = uproot.concatenate(bkg_path, library="pd")
    ak_sig = apply_selection(ak_sig, signal_selection, "signal_selection")
    ak_bkg = apply_selection(ak_bkg, background_selection, "background_selection")

    ak_sig["is_sig"] = True
    ak_bkg["is_sig"] = False
    ak_sig["is_bkg"] = False
    ak_bkg["is_bkg"] = True

    df_raw = pd.concat([ak_sig, ak_bkg], axis=0, ignore_index=True)
    scaler = StandardScaler()
    df_trans = pd.DataFrame(
        scaler.fit_transform(df_raw[input_columns]),
        columns=trans_columns,
        index=df_raw.index,
    )
    df = pd.concat([df_trans, df_raw], axis=1)

    X = df[trans_columns]
    y = df[["is_sig", "is_bkg"]]
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y["is_sig"],
        random_state=CORE_PARAMS["random_state"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest,
        y_valtest,
        test_size=0.5,
        stratify=y_valtest["is_sig"],
        random_state=CORE_PARAMS["random_state"],
    )

    stage_cfg = STAGE_CONFIGS[stage_group]
    n_sig_train = int(y_train["is_sig"].sum())
    n_bkg_train = int(y_train["is_bkg"].sum())
    base_scale_pos_weight = float(n_bkg_train / max(n_sig_train, 1))
    print(f"Training sample ratio n_bkg/n_sig = {base_scale_pos_weight:.6f}")
    current_params = deepcopy(stage_cfg["baseline"])
    if current_params.get("scale_pos_weight") == "ratio":
        current_params["scale_pos_weight"] = base_scale_pos_weight
    completed_steps = []
    stage_summaries = {}

    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        current_params = state.get("current_params", current_params)
        completed_steps = state.get("completed_steps", [])
        stage_summaries = state.get("stage_summaries", {})
        print(f"Resuming from state: completed={completed_steps}")

    ordered_steps = ["step1", "step2", "step3", "step4", "step5", "step6"]
    for step_name in ordered_steps:
        if step_name in completed_steps:
            print(f"Skipping {step_name} (already completed)")
            continue

        if step_name == "step6":
            search_space = build_step6_space(current_params)
        else:
            search_space = deepcopy(stage_cfg[step_name])
            if step_name == "step5" and "scale_pos_weight" in search_space:
                spw_cfg = search_space["scale_pos_weight"]
                search_space["scale_pos_weight"]["low"] = float(spw_cfg["low"]) * base_scale_pos_weight
                search_space["scale_pos_weight"]["high"] = float(spw_cfg["high"]) * base_scale_pos_weight

        print(f"\n=== Running {step_name} ===")
        print(json.dumps(search_space, indent=2))

        current_params, summary = run_stage(
            step_name=step_name,
            search_space=search_space,
            current_params=current_params,
            n_trials=n_trials,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        stage_summaries[step_name] = summary
        with open(os.path.join(stages_dir, f"{step_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(stages_dir, f"{step_name}_best_params.json"), "w") as f:
            json.dump(summary["updated_params"], f, indent=2)

        completed_steps.append(step_name)
        with open(state_path, "w") as f:
            json.dump(
                {
                    "train_tag": train_tag,
                    "stage_group_requested": stage_group_requested,
                    "stage_group": stage_group,
                    "completed_steps": completed_steps,
                    "current_params": current_params,
                    "stage_summaries": stage_summaries,
                },
                f,
                indent=2,
            )

    final_params = {**CORE_PARAMS, **current_params}
    print("\n=== Final params after Step6 ===")
    print(json.dumps(final_params, indent=2))

    xgbc = XGBClassifier(**final_params)
    xgbc.fit(X_train, y_train["is_sig"])

    test_score_xgb = xgbc.predict_proba(X_test)
    test_score_xgb_sig = test_score_xgb[y_test["is_sig"]][:, 1]
    test_score_xgb_bkg = test_score_xgb[y_test["is_bkg"]][:, 1]
    test_score_xgb_all = test_score_xgb[:, 1]
    test_y_true = y_test["is_sig"].astype(int).to_numpy()

    score_plot_path = condor_training_score_path(train_tag)
    plt.figure(figsize=(6, 6))
    plt.hist(test_score_xgb_sig, label="X(3872)", histtype="step", bins=np.linspace(0, 1, 100), density=True)
    plt.hist(test_score_xgb_bkg, label="bkg", histtype="step", bins=np.linspace(0, 1, 100), density=True)
    plt.xlabel("Score (Prob. from XGBoost Prediction)")
    plt.ylabel("(Bin Width)$^{-1}$")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(score_plot_path)
    plt.close()

    fpr, tpr, thresholds = roc_curve(test_y_true, test_score_xgb_all)
    roc_auc = auc(fpr, tpr)
    background_rejection = 1.0 - fpr

    roc_json_path = os.path.join(condor_training_dir(train_tag), "test_roc.json")
    with open(roc_json_path, "w") as f:
        json.dump(
            {
                "auc": float(roc_auc),
                "threshold": [float(x) for x in thresholds],
                "tpr": [float(x) for x in tpr],
                "fpr": [float(x) for x in fpr],
                "background_rejection": [float(x) for x in background_rejection],
                "signal_efficiency": [float(x) for x in tpr],
            },
            f,
            indent=2,
        )

    roc_plot_path = os.path.join(condor_training_dir(train_tag), "test_roc.pdf")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Background efficiency (FPR)")
    plt.ylabel("Signal efficiency (TPR)")
    plt.title(f"{train_tag} | Test ROC")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()

    trained_model_path = condor_model_path(train_tag)
    joblib.dump(xgbc, trained_model_path)

    trained_scaler_path = condor_scaler_path(train_tag)
    joblib.dump(scaler, trained_scaler_path)

    config_output_path = condor_model_config_path(train_tag)
    with open(config_output_path, "w") as f:
        json.dump({"input_columns": input_columns, "trans_columns": trans_columns}, f, indent=2)

    save_feature_importance(xgbc, input_columns, train_tag)

    tuned_keys = sorted(set().union(*[stage_cfg[name].keys() for name in ["step1", "step2", "step3", "step4", "step5"] + []]))
    tuned_keys.extend([k for k in ["max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree"] if k not in tuned_keys])

    save_run_metadata(
        train_tag=train_tag,
        training_script="staged_optuna_pipeline.py",
        signal_path=sig_path,
        background_path=bkg_path,
        signal_selection=signal_selection,
        background_selection=background_selection,
        input_columns=input_columns,
        trans_columns=trans_columns,
        pos_weight=float(current_params["scale_pos_weight"]),
        fixed_model_params={**CORE_PARAMS, **stage_cfg["baseline"]},
        best_model_params=final_params,
        is_optuna=True,
        optuna_n_trials=n_trials,
        optimized_hyperparameters=sorted(set(tuned_keys)),
        hyperparameter_search_space={step: stage_cfg[step] for step in ["step1", "step2", "step3", "step4", "step5"]},
        optimization_metric="max validation AUC (staged Step1-6)",
        best_objective_value=float(stage_summaries["step6"]["best_value"]),
        notes={
            "training_mode": "staged_pipeline",
            "stage_group": stage_group,
            "stage_group_requested": stage_group_requested,
            "sample_pos_weight_ratio": base_scale_pos_weight,
            "state_path": state_path,
            "stages_dir": stages_dir,
            "completed_steps": completed_steps,
            "test_auc": float(roc_auc),
            "test_roc_json_path": roc_json_path,
            "test_roc_plot_path": roc_plot_path,
        },
    )

    print("Staged pipeline complete.")


if __name__ == "__main__":
    main()
