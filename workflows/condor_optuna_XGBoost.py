import argparse
import json
import os
import sys

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

from configs.optuna_spaces import OPTUNA_SPACES, OPTUNA_TRAINING_OPTIONS
from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_token_from_tag,
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
from utils.tagging import parse_optuna_spec_from_train_tag
from utils.varsets import get_varset_columns, infer_sample_from_tag, infer_varset_from_tag


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mainline Condor Optuna training with a single search space.",
    )
    parser.add_argument("train_tag", help="Full training tag")
    parser.add_argument(
        "legacy_search_space_tag",
        nargs="?",
        default="",
        help="Deprecated legacy argument; ignored in single-space mode.",
    )
    parser.add_argument(
        "--dataset-year",
        choices=["2023", "2024"],
        default=None,
        help="Optional override for PbPb training dataset year",
    )
    parser.add_argument(
        "--selection-profile",
        default=None,
        help="Optional override for training selection profile",
    )
    return parser.parse_args()


args = parse_args()
train_tag = args.train_tag
legacy_search_space_tag = args.legacy_search_space_tag
number_trials = int(os.environ.get("OPTUNA_N_TRIALS", "100"))

def resolve_training_inputs():
    sample = infer_sample_from_config(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset_year = args.dataset_year or infer_dataset_year(train_tag, sample)
    profile_key = args.selection_profile or infer_selection_profile(train_tag, sample)
    training_cfg = resolve_training_config(sample, channel, dataset_year, profile_key)
    return (
        sample,
        channel,
        dataset_year,
        training_cfg["dataset_source"],
        profile_key,
        to_root_spec(training_cfg["signal"]),
        to_root_spec(training_cfg["background"]),
        training_cfg["signal_selection"],
        training_cfg["background_selection"],
    )


SAMPLE_KEY, CHANNEL, DATASET_YEAR, DATASET_SOURCE, SELECTION_PROFILE, SIG_PATH, BKG_PATH, SIGNAL_SELECTION, BACKGROUND_SELECTION = resolve_training_inputs()
DATASET_TOKEN = infer_dataset_token_from_tag(train_tag)
OPTUNA_OBJECTIVE_INDEX, OPTUNA_N_TRIALS_FROM_TAG, OPTUNA_SPACE_VERSION = parse_optuna_spec_from_train_tag(train_tag)
if number_trials != OPTUNA_N_TRIALS_FROM_TAG:
    print(
        f"Warning: OPTUNA_N_TRIALS env ({number_trials}) != tag trials ({OPTUNA_N_TRIALS_FROM_TAG}). "
        f"Using tag value: {OPTUNA_N_TRIALS_FROM_TAG}"
    )
    number_trials = OPTUNA_N_TRIALS_FROM_TAG

sample_key = infer_sample_from_tag(train_tag)
feature_set_tag = infer_varset_from_tag(train_tag, sample=sample_key)
if feature_set_tag is None:
    raise ValueError(f"Unable to infer feature set from train_tag: {train_tag}")
input_columns = get_varset_columns(sample_key, feature_set_tag, channel=CHANNEL)
FIXED_MODEL_PARAMS = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def int_range(low, high):
    return {"type": "int", "low": low, "high": high}


def float_range(low, high, log=False):
    config = {"type": "float", "low": low, "high": high}
    if log:
        config["log"] = True
    return config


SEARCH_SPACE_PRESETS = {
    "v1": {"n_estimators": int_range(450, 1200), "learning_rate": float_range(0.02, 0.06, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(8, 18), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(1.0, 4.0), "reg_alpha": float_range(0.5, 2.5), "reg_lambda": float_range(3.0, 9.0), "max_delta_step": float_range(0.0, 2.0)},
    "v2": {"n_estimators": int_range(900, 2200), "learning_rate": float_range(0.008, 0.025, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(8, 20), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(1.5, 4.5), "reg_alpha": float_range(0.8, 3.0), "reg_lambda": float_range(4.0, 10.0), "max_delta_step": float_range(0.0, 2.0)},
    "v3": {"n_estimators": int_range(350, 900), "learning_rate": float_range(0.05, 0.10), "max_depth": int_range(2, 3), "min_child_weight": int_range(6, 14), "subsample": float_range(0.80, 1.00), "colsample_bytree": float_range(0.80, 1.00), "gamma": float_range(0.8, 3.0), "reg_alpha": float_range(0.2, 1.5), "reg_lambda": float_range(2.0, 7.0), "max_delta_step": float_range(0.0, 1.5)},
    "v4": {"n_estimators": int_range(500, 1400), "learning_rate": float_range(0.015, 0.04, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(10, 24), "subsample": float_range(0.60, 0.80), "colsample_bytree": float_range(0.60, 0.80), "gamma": float_range(2.0, 6.0), "reg_alpha": float_range(1.0, 4.0), "reg_lambda": float_range(5.0, 14.0), "max_delta_step": float_range(0.0, 3.0)},
    "v5": {"n_estimators": int_range(400, 1100), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(3, 4), "min_child_weight": int_range(6, 16), "subsample": float_range(0.85, 1.00), "colsample_bytree": float_range(0.85, 1.00), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.2, 1.2), "reg_lambda": float_range(2.0, 6.0), "max_delta_step": float_range(0.0, 1.0)},
    "v6": {"n_estimators": int_range(600, 1800), "learning_rate": float_range(0.01, 0.03, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(12, 28), "subsample": float_range(0.55, 0.75), "colsample_bytree": float_range(0.55, 0.75), "gamma": float_range(2.5, 7.0), "reg_alpha": float_range(1.5, 5.0), "reg_lambda": float_range(6.0, 16.0), "max_delta_step": float_range(1.0, 4.0)},
    "v7": {"n_estimators": int_range(500, 1500), "learning_rate": float_range(0.02, 0.07, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(6, 18), "subsample": float_range(0.65, 0.85), "colsample_bytree": float_range(0.90, 1.00), "gamma": float_range(0.8, 3.0), "reg_alpha": float_range(0.4, 2.0), "reg_lambda": float_range(2.0, 8.0), "max_delta_step": float_range(0.0, 2.0)},
    "v8": {"n_estimators": int_range(550, 1600), "learning_rate": float_range(0.02, 0.06, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(8, 18), "subsample": float_range(0.90, 1.00), "colsample_bytree": float_range(0.60, 0.80), "gamma": float_range(1.0, 4.0), "reg_alpha": float_range(0.5, 2.5), "reg_lambda": float_range(3.0, 10.0), "max_delta_step": float_range(0.0, 2.0)},
    "v9": {"n_estimators": int_range(300, 800), "learning_rate": float_range(0.08, 0.16), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 10), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(0.3, 2.0), "reg_alpha": float_range(0.0, 1.0), "reg_lambda": float_range(1.0, 5.0), "max_delta_step": float_range(0.0, 1.0)},
    "v10": {"n_estimators": int_range(400, 2000), "learning_rate": float_range(0.01, 0.12, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(4, 24), "subsample": float_range(0.55, 1.00), "colsample_bytree": float_range(0.55, 1.00), "gamma": float_range(0.0, 6.0), "reg_alpha": float_range(0.0, 4.0), "reg_lambda": float_range(1.0, 14.0), "max_delta_step": float_range(0.0, 4.0)},
    "v11": {"n_estimators": int_range(1200, 3000), "learning_rate": float_range(0.006, 0.018, log=True), "max_depth": int_range(3, 4), "min_child_weight": int_range(4, 12), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.2, 1.5), "reg_lambda": float_range(2.0, 8.0), "max_delta_step": float_range(0.0, 2.0)},
    "v12": {"n_estimators": int_range(1600, 3400), "learning_rate": float_range(0.004, 0.012, log=True), "max_depth": int_range(3, 4), "min_child_weight": int_range(6, 14), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(0.8, 3.0), "reg_alpha": float_range(0.5, 2.0), "reg_lambda": float_range(3.0, 10.0), "max_delta_step": float_range(0.0, 2.5)},
    "v13": {"n_estimators": int_range(900, 2200), "learning_rate": float_range(0.01, 0.025, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(3, 10), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(0.2, 1.8), "reg_alpha": float_range(0.0, 1.2), "reg_lambda": float_range(1.0, 6.0), "max_delta_step": float_range(0.0, 1.5)},
    "v14": {"n_estimators": int_range(1000, 2600), "learning_rate": float_range(0.008, 0.022, log=True), "max_depth": int_range(4, 5), "min_child_weight": int_range(4, 10), "subsample": float_range(0.80, 1.00), "colsample_bytree": float_range(0.80, 1.00), "gamma": float_range(0.0, 1.5), "reg_alpha": float_range(0.0, 1.0), "reg_lambda": float_range(1.0, 5.0), "max_delta_step": float_range(0.0, 1.0)},
    "v15": {"n_estimators": int_range(1400, 3200), "learning_rate": float_range(0.005, 0.016, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(8, 18), "subsample": float_range(0.60, 0.80), "colsample_bytree": float_range(0.60, 0.80), "gamma": float_range(1.2, 4.0), "reg_alpha": float_range(0.8, 3.0), "reg_lambda": float_range(4.0, 12.0), "max_delta_step": float_range(0.5, 3.0)},
    "v16": {"n_estimators": int_range(700, 1800), "learning_rate": float_range(0.015, 0.04, log=True), "max_depth": int_range(3, 4), "min_child_weight": int_range(4, 10), "subsample": float_range(0.90, 1.00), "colsample_bytree": float_range(0.90, 1.00), "gamma": float_range(0.2, 1.5), "reg_alpha": float_range(0.0, 0.8), "reg_lambda": float_range(1.0, 5.0), "max_delta_step": float_range(0.0, 1.0)},
    "v17": {"n_estimators": int_range(1100, 2600), "learning_rate": float_range(0.007, 0.020, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(2, 8), "subsample": float_range(0.65, 0.85), "colsample_bytree": float_range(0.65, 0.85), "gamma": float_range(0.0, 1.5), "reg_alpha": float_range(0.0, 1.0), "reg_lambda": float_range(1.0, 6.0), "max_delta_step": float_range(0.0, 1.5)},
    "v18": {"n_estimators": int_range(1300, 3000), "learning_rate": float_range(0.006, 0.018, log=True), "max_depth": int_range(4, 5), "min_child_weight": int_range(6, 14), "subsample": float_range(0.85, 1.00), "colsample_bytree": float_range(0.60, 0.80), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.2, 1.5), "reg_lambda": float_range(2.0, 8.0), "max_delta_step": float_range(0.0, 2.0)},
    "v19": {"n_estimators": int_range(900, 2400), "learning_rate": float_range(0.01, 0.03, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(10, 22), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(1.5, 5.0), "reg_alpha": float_range(1.0, 4.0), "reg_lambda": float_range(4.0, 12.0), "max_delta_step": float_range(1.0, 4.0)},
    "v20": {"n_estimators": int_range(700, 3200), "learning_rate": float_range(0.004, 0.035, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(2, 18), "subsample": float_range(0.60, 1.00), "colsample_bytree": float_range(0.60, 1.00), "gamma": float_range(0.0, 4.0), "reg_alpha": float_range(0.0, 3.0), "reg_lambda": float_range(1.0, 12.0), "max_delta_step": float_range(0.0, 4.0)},
    "v21": {"n_estimators": int_range(350, 1100), "learning_rate": float_range(0.04, 0.10), "max_depth": int_range(4, 5), "min_child_weight": int_range(2, 8), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(0.0, 1.2), "reg_alpha": float_range(0.0, 0.8), "reg_lambda": float_range(1.0, 4.0), "max_delta_step": float_range(0.0, 1.0)},
    "v22": {"n_estimators": int_range(500, 1500), "learning_rate": float_range(0.025, 0.07), "max_depth": int_range(5, 6), "min_child_weight": int_range(1, 5), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(0.0, 1.0), "reg_alpha": float_range(0.0, 0.8), "reg_lambda": float_range(1.0, 4.0), "max_delta_step": float_range(0.0, 1.0)},
    "v23": {"n_estimators": int_range(250, 850), "learning_rate": float_range(0.08, 0.16), "max_depth": int_range(5, 7), "min_child_weight": int_range(1, 4), "subsample": float_range(0.80, 1.00), "colsample_bytree": float_range(0.80, 1.00), "gamma": float_range(0.0, 0.8), "reg_alpha": float_range(0.0, 0.6), "reg_lambda": float_range(1.0, 3.5), "max_delta_step": float_range(0.0, 0.5)},
    "v24": {"n_estimators": int_range(700, 1800), "learning_rate": float_range(0.015, 0.04, log=True), "max_depth": int_range(5, 6), "min_child_weight": int_range(1, 5), "subsample": float_range(0.85, 1.00), "colsample_bytree": float_range(0.85, 1.00), "gamma": float_range(0.0, 1.0), "reg_alpha": float_range(0.0, 0.8), "reg_lambda": float_range(1.0, 4.5), "max_delta_step": float_range(0.0, 1.0)},
    "v25": {"n_estimators": int_range(450, 1400), "learning_rate": float_range(0.03, 0.09), "max_depth": int_range(4, 6), "min_child_weight": int_range(2, 6), "subsample": float_range(0.90, 1.00), "colsample_bytree": float_range(0.60, 0.80), "gamma": float_range(0.0, 1.5), "reg_alpha": float_range(0.0, 1.0), "reg_lambda": float_range(1.0, 5.0), "max_delta_step": float_range(0.0, 1.5)},
    "v26": {"n_estimators": int_range(300, 900), "learning_rate": float_range(0.10, 0.20), "max_depth": int_range(4, 6), "min_child_weight": int_range(1, 4), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(0.0, 1.0), "reg_alpha": float_range(0.0, 0.8), "reg_lambda": float_range(1.0, 4.0), "max_delta_step": float_range(0.0, 0.5)},
    "v27": {"n_estimators": int_range(600, 1800), "learning_rate": float_range(0.015, 0.05, log=True), "max_depth": int_range(5, 7), "min_child_weight": int_range(4, 10), "subsample": float_range(0.55, 0.75), "colsample_bytree": float_range(0.55, 0.75), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.2, 1.5), "reg_lambda": float_range(2.0, 7.0), "max_delta_step": float_range(0.0, 2.0)},
    "v28": {"n_estimators": int_range(250, 750), "learning_rate": float_range(0.12, 0.24), "max_depth": int_range(6, 8), "min_child_weight": int_range(1, 3), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(0.0, 0.6), "reg_alpha": float_range(0.0, 0.5), "reg_lambda": float_range(1.0, 3.0), "max_delta_step": float_range(0.0, 0.5)},
    "v29": {"n_estimators": int_range(800, 2200), "learning_rate": float_range(0.01, 0.03, log=True), "max_depth": int_range(4, 6), "min_child_weight": int_range(2, 8), "subsample": float_range(0.90, 1.00), "colsample_bytree": float_range(0.90, 1.00), "gamma": float_range(0.0, 1.2), "reg_alpha": float_range(0.0, 0.8), "reg_lambda": float_range(1.0, 4.5), "max_delta_step": float_range(0.0, 1.0)},
    "v30": {"n_estimators": int_range(300, 1800), "learning_rate": float_range(0.01, 0.18, log=True), "max_depth": int_range(4, 7), "min_child_weight": int_range(1, 12), "subsample": float_range(0.55, 1.00), "colsample_bytree": float_range(0.55, 1.00), "gamma": float_range(0.0, 3.0), "reg_alpha": float_range(0.0, 2.0), "reg_lambda": float_range(1.0, 8.0), "max_delta_step": float_range(0.0, 3.0)},
    "v31": {"n_estimators": int_range(600, 1800), "learning_rate": float_range(0.015, 0.05, log=True), "max_depth": int_range(5, 6), "min_child_weight": int_range(10, 22), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(1.5, 5.0), "reg_alpha": float_range(1.0, 4.0), "reg_lambda": float_range(4.0, 12.0), "max_delta_step": float_range(1.0, 4.0)},
    "v32": {"n_estimators": int_range(350, 1100), "learning_rate": float_range(0.04, 0.10), "max_depth": int_range(6, 7), "min_child_weight": int_range(8, 18), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(1.0, 4.0), "reg_alpha": float_range(0.8, 3.0), "reg_lambda": float_range(3.0, 10.0), "max_delta_step": float_range(1.0, 3.0)},
    "v33": {"n_estimators": int_range(900, 2600), "learning_rate": float_range(0.008, 0.025, log=True), "max_depth": int_range(5, 7), "min_child_weight": int_range(6, 14), "subsample": float_range(0.80, 1.00), "colsample_bytree": float_range(0.80, 1.00), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.2, 1.5), "reg_lambda": float_range(2.0, 8.0), "max_delta_step": float_range(0.0, 2.0)},
    "v34": {"n_estimators": int_range(300, 900), "learning_rate": float_range(0.06, 0.16), "max_depth": int_range(6, 8), "min_child_weight": int_range(12, 24), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(2.0, 6.0), "reg_alpha": float_range(1.5, 5.0), "reg_lambda": float_range(5.0, 14.0), "max_delta_step": float_range(2.0, 6.0)},
    "v35": {"n_estimators": int_range(600, 1800), "learning_rate": float_range(0.015, 0.045, log=True), "max_depth": int_range(5, 7), "min_child_weight": int_range(14, 30), "subsample": float_range(0.85, 1.00), "colsample_bytree": float_range(0.85, 1.00), "gamma": float_range(2.0, 6.0), "reg_alpha": float_range(0.5, 2.0), "reg_lambda": float_range(6.0, 18.0), "max_delta_step": float_range(2.0, 8.0)},
    "v36": {"n_estimators": int_range(450, 1400), "learning_rate": float_range(0.02, 0.07, log=True), "max_depth": int_range(4, 6), "min_child_weight": int_range(16, 34), "subsample": float_range(0.60, 0.80), "colsample_bytree": float_range(0.60, 0.80), "gamma": float_range(2.5, 7.0), "reg_alpha": float_range(1.5, 5.0), "reg_lambda": float_range(6.0, 18.0), "max_delta_step": float_range(3.0, 8.0)},
    "v37": {"n_estimators": int_range(500, 1600), "learning_rate": float_range(0.02, 0.08), "max_depth": int_range(5, 6), "min_child_weight": int_range(8, 18), "subsample": float_range(0.90, 1.00), "colsample_bytree": float_range(0.55, 0.75), "gamma": float_range(1.0, 4.0), "reg_alpha": float_range(0.8, 3.0), "reg_lambda": float_range(3.0, 10.0), "max_delta_step": float_range(1.0, 4.0)},
    "v38": {"n_estimators": int_range(700, 2200), "learning_rate": float_range(0.01, 0.03, log=True), "max_depth": int_range(6, 8), "min_child_weight": int_range(10, 20), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95), "gamma": float_range(1.5, 5.0), "reg_alpha": float_range(1.0, 4.0), "reg_lambda": float_range(4.0, 14.0), "max_delta_step": float_range(2.0, 6.0)},
    "v39": {"n_estimators": int_range(250, 800), "learning_rate": float_range(0.10, 0.22), "max_depth": int_range(5, 7), "min_child_weight": int_range(8, 18), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90), "gamma": float_range(1.5, 4.5), "reg_alpha": float_range(1.0, 4.0), "reg_lambda": float_range(4.0, 12.0), "max_delta_step": float_range(2.0, 6.0)},
    "v40": {"n_estimators": int_range(250, 2200), "learning_rate": float_range(0.008, 0.16, log=True), "max_depth": int_range(5, 8), "min_child_weight": int_range(6, 30), "subsample": float_range(0.60, 1.00), "colsample_bytree": float_range(0.55, 1.00), "gamma": float_range(0.5, 7.0), "reg_alpha": float_range(0.2, 5.0), "reg_lambda": float_range(2.0, 18.0), "max_delta_step": float_range(0.0, 8.0)},
    "v41": {"n_estimators": int_range(500, 1600), "learning_rate": float_range(0.015, 0.05, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(2, 8), "subsample": float_range(0.40, 0.60), "colsample_bytree": float_range(0.40, 0.60), "colsample_bylevel": float_range(0.40, 0.70), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.3, 2.0), "reg_lambda": float_range(2.0, 8.0), "max_delta_step": float_range(0.0, 2.0)},
    "v42": {"n_estimators": int_range(600, 1800), "learning_rate": float_range(0.01, 0.04, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(2, 8), "subsample": float_range(0.85, 1.00), "colsample_bytree": float_range(0.35, 0.55), "colsample_bylevel": float_range(0.35, 0.60), "gamma": float_range(0.0, 2.0), "reg_alpha": float_range(0.0, 1.5), "reg_lambda": float_range(1.0, 6.0), "max_delta_step": float_range(0.0, 1.5)},
    "v43": {"n_estimators": int_range(600, 1800), "learning_rate": float_range(0.015, 0.05, log=True), "max_depth": int_range(4, 6), "min_child_weight": int_range(3, 10), "subsample": float_range(0.35, 0.55), "colsample_bytree": float_range(0.80, 1.00), "colsample_bylevel": float_range(0.80, 1.00), "gamma": float_range(0.0, 1.5), "reg_alpha": float_range(0.0, 1.0), "reg_lambda": float_range(1.0, 5.0), "max_delta_step": float_range(0.0, 1.0)},
    "v44": {"n_estimators": int_range(900, 2400), "learning_rate": float_range(0.008, 0.025, log=True), "max_depth": int_range(3, 5), "min_child_weight": int_range(6, 14), "subsample": float_range(0.45, 0.70), "colsample_bytree": float_range(0.45, 0.70), "colsample_bylevel": float_range(0.45, 0.70), "gamma": float_range(0.8, 3.0), "reg_alpha": float_range(0.5, 2.5), "reg_lambda": float_range(3.0, 9.0), "max_delta_step": float_range(0.0, 3.0)},
    "v45": {"n_estimators": int_range(350, 1100), "learning_rate": float_range(0.05, 0.12), "max_depth": int_range(4, 6), "min_child_weight": int_range(4, 12), "subsample": float_range(0.45, 0.65), "colsample_bytree": float_range(0.45, 0.65), "colsample_bylevel": float_range(0.45, 0.65), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.5, 2.0), "reg_lambda": float_range(2.0, 7.0), "max_delta_step": float_range(0.0, 2.0)},
    "v46": {"n_estimators": int_range(450, 1400), "learning_rate": float_range(0.02, 0.07, log=True), "max_depth": int_range(5, 7), "min_child_weight": int_range(6, 14), "subsample": float_range(0.35, 0.55), "colsample_bytree": float_range(0.35, 0.55), "colsample_bylevel": float_range(0.35, 0.55), "gamma": float_range(1.0, 3.5), "reg_alpha": float_range(0.5, 2.5), "reg_lambda": float_range(2.0, 8.0), "max_delta_step": float_range(0.0, 3.0)},
    "v47": {"n_estimators": int_range(700, 2000), "learning_rate": float_range(0.01, 0.03, log=True), "max_depth": int_range(4, 6), "min_child_weight": int_range(10, 22), "subsample": float_range(0.50, 0.75), "colsample_bytree": float_range(0.85, 1.00), "colsample_bylevel": float_range(0.40, 0.70), "gamma": float_range(1.0, 4.0), "reg_alpha": float_range(1.0, 3.5), "reg_lambda": float_range(4.0, 12.0), "max_delta_step": float_range(1.0, 4.0)},
    "v48": {"n_estimators": int_range(300, 900), "learning_rate": float_range(0.08, 0.18), "max_depth": int_range(3, 5), "min_child_weight": int_range(2, 8), "subsample": float_range(0.30, 0.50), "colsample_bytree": float_range(0.80, 1.00), "colsample_bylevel": float_range(0.80, 1.00), "gamma": float_range(0.0, 1.5), "reg_alpha": float_range(0.0, 1.2), "reg_lambda": float_range(1.0, 5.0), "max_delta_step": float_range(0.0, 1.5)},
    "v49": {"n_estimators": int_range(800, 2200), "learning_rate": float_range(0.008, 0.025, log=True), "max_depth": int_range(4, 6), "min_child_weight": int_range(4, 12), "subsample": float_range(0.80, 1.00), "colsample_bytree": float_range(0.30, 0.45), "colsample_bylevel": float_range(0.30, 0.45), "gamma": float_range(0.5, 2.5), "reg_alpha": float_range(0.2, 1.5), "reg_lambda": float_range(2.0, 7.0), "max_delta_step": float_range(0.0, 2.0)},
    "v50": {"n_estimators": int_range(300, 2200), "learning_rate": float_range(0.008, 0.18, log=True), "max_depth": int_range(3, 7), "min_child_weight": int_range(2, 22), "subsample": float_range(0.30, 1.00), "colsample_bytree": float_range(0.30, 1.00), "colsample_bylevel": float_range(0.30, 1.00), "gamma": float_range(0.0, 5.0), "reg_alpha": float_range(0.0, 4.0), "reg_lambda": float_range(1.0, 14.0), "max_delta_step": float_range(0.0, 6.0)},
    "v51": {"n_estimators": int_range(1800, 2200), "learning_rate": float_range(0.015, 0.020, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(22, 28), "subsample": float_range(0.50, 0.56), "colsample_bytree": float_range(0.50, 0.56), "gamma": float_range(3.8, 4.8), "reg_alpha": float_range(4.8, 5.8), "reg_lambda": float_range(12.0, 14.5), "max_delta_step": float_range(0.0, 1.5)},
    "v52": {"n_estimators": int_range(2000, 2200), "learning_rate": float_range(0.017, 0.020, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(24, 27), "subsample": float_range(0.51, 0.54), "colsample_bytree": float_range(0.51, 0.54), "gamma": float_range(3.9, 4.4), "reg_alpha": float_range(5.0, 5.6), "reg_lambda": float_range(12.5, 13.8), "max_delta_step": float_range(0.0, 1.0)},
    "v53": {"n_estimators": int_range(1800, 2200), "learning_rate": float_range(0.014, 0.019, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(23, 28), "subsample": float_range(0.48, 0.54), "colsample_bytree": float_range(0.48, 0.54), "gamma": float_range(3.6, 4.6), "reg_alpha": float_range(4.8, 6.0), "reg_lambda": float_range(12.0, 15.0), "max_delta_step": float_range(0.0, 1.5)},
    "v54": {"n_estimators": int_range(1900, 2200), "learning_rate": float_range(0.016, 0.020, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(22, 28), "subsample": float_range(0.50, 0.58), "colsample_bytree": float_range(0.50, 0.58), "gamma": float_range(3.8, 4.8), "reg_alpha": float_range(5.0, 6.0), "reg_lambda": float_range(12.0, 15.0), "max_delta_step": float_range(0.0, 2.0)},
    "v55": {"n_estimators": int_range(1700, 2200), "learning_rate": float_range(0.012, 0.018, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(20, 26), "subsample": float_range(0.50, 0.60), "colsample_bytree": float_range(0.50, 0.60), "gamma": float_range(3.5, 5.0), "reg_alpha": float_range(4.5, 6.0), "reg_lambda": float_range(11.5, 15.5), "max_delta_step": float_range(0.0, 2.0)},
    "v56": {"n_estimators": int_range(2000, 2200), "learning_rate": float_range(0.015, 0.020, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(24, 28), "subsample": float_range(0.50, 0.54), "colsample_bytree": float_range(0.50, 0.54), "gamma": float_range(3.9, 4.3), "reg_alpha": float_range(5.1, 5.8), "reg_lambda": float_range(13.0, 14.5), "max_delta_step": float_range(0.5, 2.5)},
    "v57": {"n_estimators": int_range(1800, 2200), "learning_rate": float_range(0.016, 0.020, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(23, 28), "subsample": float_range(0.49, 0.55), "colsample_bytree": float_range(0.49, 0.55), "gamma": float_range(4.0, 5.2), "reg_alpha": float_range(5.0, 6.2), "reg_lambda": float_range(12.0, 14.8), "max_delta_step": float_range(0.0, 1.5)},
    "v58": {"n_estimators": int_range(1500, 2200), "learning_rate": float_range(0.010, 0.018, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(22, 30), "subsample": float_range(0.45, 0.60), "colsample_bytree": float_range(0.45, 0.60), "gamma": float_range(3.5, 5.5), "reg_alpha": float_range(4.5, 6.5), "reg_lambda": float_range(12.0, 16.0), "max_delta_step": float_range(0.0, 3.0)},
    "v59": {"n_estimators": int_range(1900, 2200), "learning_rate": float_range(0.017, 0.020, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(24, 26), "subsample": float_range(0.515, 0.535), "colsample_bytree": float_range(0.515, 0.535), "gamma": float_range(4.0, 4.3), "reg_alpha": float_range(5.1, 5.5), "reg_lambda": float_range(13.0, 13.6), "max_delta_step": float_range(0.0, 1.0)},
    "v60": {"n_estimators": int_range(1700, 2200), "learning_rate": float_range(0.014, 0.020, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(20, 30), "subsample": float_range(0.45, 0.60), "colsample_bytree": float_range(0.45, 0.60), "gamma": float_range(3.5, 5.5), "reg_alpha": float_range(4.5, 6.5), "reg_lambda": float_range(11.5, 16.0), "max_delta_step": float_range(0.0, 3.0)},
    "v61": {"n_estimators": int_range(500, 900), "learning_rate": float_range(0.05, 0.10), "max_depth": int_range(2, 2), "min_child_weight": int_range(4, 8), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v62": {"n_estimators": int_range(700, 1200), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 8), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v63": {"n_estimators": int_range(900, 1500), "learning_rate": float_range(0.02, 0.06, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(5, 10), "subsample": float_range(0.70, 0.88), "colsample_bytree": float_range(0.70, 0.88)},
    "v64": {"n_estimators": int_range(600, 1100), "learning_rate": float_range(0.04, 0.09), "max_depth": int_range(3, 3), "min_child_weight": int_range(3, 7), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95)},
    "v65": {"n_estimators": int_range(800, 1400), "learning_rate": float_range(0.025, 0.06, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(4, 8), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v66": {"n_estimators": int_range(1000, 1700), "learning_rate": float_range(0.015, 0.04, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(5, 10), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95)},
    "v67": {"n_estimators": int_range(700, 1300), "learning_rate": float_range(0.03, 0.07), "max_depth": int_range(2, 4), "min_child_weight": int_range(4, 9), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v68": {"n_estimators": int_range(900, 1600), "learning_rate": float_range(0.02, 0.05, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(5, 10), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95)},
    "v69": {"n_estimators": int_range(1100, 1800), "learning_rate": float_range(0.012, 0.03, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(6, 10), "subsample": float_range(0.70, 0.88), "colsample_bytree": float_range(0.70, 0.88)},
    "v70": {"n_estimators": int_range(600, 1400), "learning_rate": float_range(0.03, 0.10), "max_depth": int_range(2, 4), "min_child_weight": int_range(3, 10), "subsample": float_range(0.65, 0.95), "colsample_bytree": float_range(0.65, 0.95)},
    "v71": {"n_estimators": int_range(1200, 2200), "learning_rate": float_range(0.010, 0.025, log=True), "max_depth": int_range(2, 2), "min_child_weight": int_range(4, 8), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v72": {"n_estimators": int_range(1500, 2600), "learning_rate": float_range(0.008, 0.020, log=True), "max_depth": int_range(2, 2), "min_child_weight": int_range(5, 9), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95)},
    "v73": {"n_estimators": int_range(1800, 3200), "learning_rate": float_range(0.006, 0.015, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(6, 10), "subsample": float_range(0.80, 0.95), "colsample_bytree": float_range(0.80, 0.95)},
    "v74": {"n_estimators": int_range(900, 1600), "learning_rate": float_range(0.015, 0.035, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 8), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v75": {"n_estimators": int_range(700, 1300), "learning_rate": float_range(0.025, 0.050, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 8), "subsample": float_range(0.70, 0.88), "colsample_bytree": float_range(0.70, 0.88)},
    "v76": {"n_estimators": int_range(500, 1000), "learning_rate": float_range(0.040, 0.080), "max_depth": int_range(2, 3), "min_child_weight": int_range(3, 7), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v77": {"n_estimators": int_range(1000, 1800), "learning_rate": float_range(0.012, 0.030, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(4, 9), "subsample": float_range(0.75, 0.95), "colsample_bytree": float_range(0.75, 0.95)},
    "v78": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.020, 0.050, log=True), "max_depth": int_range(3, 3), "min_child_weight": int_range(4, 9), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v79": {"n_estimators": int_range(500, 900), "learning_rate": float_range(0.050, 0.090), "max_depth": int_range(3, 3), "min_child_weight": int_range(3, 7), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v80": {"n_estimators": int_range(700, 2200), "learning_rate": float_range(0.008, 0.080, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 10), "subsample": float_range(0.70, 0.95), "colsample_bytree": float_range(0.70, 0.95)},
    "v81": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 10), "subsample": float_range(0.85, 0.95), "colsample_bytree": float_range(0.85, 0.95)},
    "v82": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 10), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v83": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(2, 3), "min_child_weight": int_range(4, 10), "subsample": float_range(0.65, 0.85), "colsample_bytree": float_range(0.65, 0.85)},
    "v84": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(3, 4), "min_child_weight": int_range(4, 10), "subsample": float_range(0.80, 0.95), "colsample_bytree": float_range(0.60, 0.75)},
    "v85": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(3, 4), "min_child_weight": int_range(4, 10), "subsample": float_range(0.60, 0.75), "colsample_bytree": float_range(0.80, 0.95)},
    "v86": {"n_estimators": int_range(800, 1600), "learning_rate": float_range(0.02, 0.06, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(5, 12), "subsample": float_range(0.55, 0.70), "colsample_bytree": float_range(0.55, 0.70)},
    "v87": {"n_estimators": int_range(800, 1600), "learning_rate": float_range(0.02, 0.06, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(5, 12), "subsample": float_range(0.50, 0.65), "colsample_bytree": float_range(0.80, 0.95)},
    "v88": {"n_estimators": int_range(800, 1600), "learning_rate": float_range(0.02, 0.06, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(5, 12), "subsample": float_range(0.80, 0.95), "colsample_bytree": float_range(0.50, 0.65)},
    "v89": {"n_estimators": int_range(900, 1800), "learning_rate": float_range(0.015, 0.05, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(6, 12), "subsample": float_range(0.50, 0.80), "colsample_bytree": float_range(0.50, 0.80)},
    "v90": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.08), "max_depth": int_range(2, 4), "min_child_weight": int_range(4, 12), "subsample": float_range(0.50, 0.95), "colsample_bytree": float_range(0.50, 0.95)},
    "v91": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.07), "max_depth": int_range(2, 3), "min_child_weight": int_range(2, 6), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v92": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.07), "max_depth": int_range(3, 4), "min_child_weight": int_range(2, 6), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v93": {"n_estimators": int_range(700, 1400), "learning_rate": float_range(0.03, 0.07), "max_depth": int_range(4, 5), "min_child_weight": int_range(2, 6), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v94": {"n_estimators": int_range(800, 1600), "learning_rate": float_range(0.025, 0.06, log=True), "max_depth": int_range(2, 3), "min_child_weight": int_range(6, 10), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v95": {"n_estimators": int_range(800, 1600), "learning_rate": float_range(0.025, 0.06, log=True), "max_depth": int_range(3, 4), "min_child_weight": int_range(6, 10), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v96": {"n_estimators": int_range(800, 1600), "learning_rate": float_range(0.025, 0.06, log=True), "max_depth": int_range(4, 5), "min_child_weight": int_range(6, 10), "subsample": float_range(0.75, 0.90), "colsample_bytree": float_range(0.75, 0.90)},
    "v97": {"n_estimators": int_range(900, 1800), "learning_rate": float_range(0.02, 0.05, log=True), "max_depth": int_range(2, 4), "min_child_weight": int_range(10, 14), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v98": {"n_estimators": int_range(900, 1800), "learning_rate": float_range(0.02, 0.05, log=True), "max_depth": int_range(3, 4), "min_child_weight": int_range(12, 18), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v99": {"n_estimators": int_range(900, 1800), "learning_rate": float_range(0.02, 0.05, log=True), "max_depth": int_range(4, 5), "min_child_weight": int_range(14, 20), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
    "v100": {"n_estimators": int_range(700, 1800), "learning_rate": float_range(0.02, 0.07, log=True), "max_depth": int_range(2, 5), "min_child_weight": int_range(2, 20), "subsample": float_range(0.70, 0.90), "colsample_bytree": float_range(0.70, 0.90)},
}

# Ordered 3v1-v100 presets.
# IMPORTANT: presets are mapped strictly by the input order (1..100),
# ignoring B/C/A/K label semantics.
_PRESET_3V_ORDERED = [
    (2, 4, 1, 8, 0.020, 0.080, 300, 900, 0.70, 0.90, 0.70, 0.90),
    (2, 4, 3, 10, 0.020, 0.080, 500, 1200, 0.70, 0.90, 0.70, 0.90),
    (2, 4, 6, 15, 0.010, 0.050, 800, 1800, 0.75, 0.95, 0.75, 0.95),
    (3, 5, 1, 8, 0.020, 0.080, 300, 900, 0.65, 0.85, 0.65, 0.85),
    (3, 5, 3, 10, 0.020, 0.080, 500, 1200, 0.65, 0.85, 0.65, 0.85),
    (3, 5, 6, 15, 0.010, 0.050, 800, 1800, 0.70, 0.90, 0.70, 0.90),
    (4, 6, 1, 8, 0.030, 0.100, 300, 900, 0.65, 0.85, 0.65, 0.85),
    (4, 6, 3, 10, 0.020, 0.080, 500, 1200, 0.65, 0.85, 0.65, 0.85),
    (4, 6, 6, 15, 0.010, 0.050, 800, 1800, 0.70, 0.90, 0.70, 0.90),
    (5, 7, 1, 8, 0.030, 0.120, 300, 1000, 0.60, 0.80, 0.60, 0.80),
    (2, 5, 1, 12, 0.010, 0.080, 400, 1400, 0.70, 0.95, 0.70, 0.95),
    (2, 5, 3, 12, 0.015, 0.100, 300, 1200, 0.65, 0.90, 0.65, 0.90),
    (2, 5, 6, 20, 0.008, 0.050, 800, 2000, 0.75, 1.00, 0.75, 1.00),
    (3, 6, 1, 12, 0.015, 0.100, 300, 1400, 0.65, 0.90, 0.65, 0.90),
    (3, 6, 3, 12, 0.020, 0.120, 300, 1200, 0.65, 0.90, 0.65, 0.90),
    (3, 6, 6, 20, 0.008, 0.050, 800, 2000, 0.70, 0.95, 0.70, 0.95),
    (4, 7, 1, 12, 0.020, 0.120, 300, 1200, 0.60, 0.85, 0.60, 0.85),
    (4, 7, 3, 12, 0.015, 0.100, 400, 1400, 0.60, 0.85, 0.60, 0.85),
    (4, 7, 6, 20, 0.008, 0.050, 800, 2000, 0.65, 0.90, 0.65, 0.90),
    (5, 8, 1, 12, 0.020, 0.150, 300, 1200, 0.60, 0.80, 0.60, 0.80),
    (2, 6, 1, 15, 0.010, 0.100, 300, 1600, 0.65, 0.95, 0.65, 0.95),
    (2, 6, 3, 15, 0.015, 0.120, 300, 1400, 0.65, 0.90, 0.65, 0.90),
    (2, 6, 8, 25, 0.005, 0.040, 1000, 2500, 0.75, 1.00, 0.75, 1.00),
    (3, 7, 1, 15, 0.015, 0.120, 300, 1400, 0.60, 0.90, 0.60, 0.90),
    (3, 7, 3, 15, 0.020, 0.120, 300, 1600, 0.60, 0.90, 0.60, 0.90),
    (3, 7, 8, 25, 0.005, 0.040, 1000, 2500, 0.70, 0.95, 0.70, 0.95),
    (4, 8, 1, 15, 0.020, 0.150, 300, 1400, 0.55, 0.85, 0.55, 0.85),
    (4, 8, 3, 15, 0.015, 0.120, 400, 1600, 0.55, 0.85, 0.55, 0.85),
    (4, 8, 8, 25, 0.005, 0.040, 1000, 2500, 0.65, 0.90, 0.65, 0.90),
    (2, 8, 1, 20, 0.010, 0.100, 500, 1800, 0.65, 0.95, 0.65, 0.95),
    (2, 3, 8, 20, 0.005, 0.030, 1200, 2500, 0.80, 1.00, 0.80, 1.00),
    (2, 3, 10, 25, 0.005, 0.030, 1500, 2500, 0.85, 1.00, 0.85, 1.00),
    (2, 3, 12, 30, 0.005, 0.020, 1800, 2500, 0.85, 1.00, 0.85, 1.00),
    (2, 4, 8, 20, 0.008, 0.040, 1000, 2200, 0.80, 1.00, 0.80, 1.00),
    (2, 4, 10, 25, 0.008, 0.040, 1200, 2500, 0.80, 1.00, 0.80, 1.00),
    (2, 4, 12, 30, 0.005, 0.030, 1500, 2500, 0.85, 1.00, 0.80, 1.00),
    (3, 4, 8, 20, 0.010, 0.050, 900, 2000, 0.75, 0.95, 0.75, 0.95),
    (3, 4, 10, 25, 0.008, 0.040, 1200, 2200, 0.75, 0.95, 0.75, 0.95),
    (3, 4, 12, 30, 0.005, 0.030, 1500, 2500, 0.80, 1.00, 0.75, 0.95),
    (3, 5, 8, 20, 0.008, 0.050, 900, 1800, 0.75, 0.95, 0.75, 0.95),
    (3, 5, 10, 25, 0.008, 0.040, 1200, 2200, 0.75, 0.95, 0.75, 0.95),
    (3, 5, 12, 30, 0.005, 0.030, 1500, 2500, 0.80, 1.00, 0.80, 1.00),
    (2, 4, 15, 30, 0.005, 0.020, 1800, 2500, 0.90, 1.00, 0.85, 1.00),
    (2, 5, 12, 25, 0.005, 0.030, 1500, 2500, 0.85, 1.00, 0.80, 1.00),
    (2, 5, 15, 30, 0.005, 0.020, 1800, 2500, 0.85, 1.00, 0.85, 1.00),
    (3, 5, 15, 30, 0.005, 0.025, 1800, 2500, 0.80, 1.00, 0.80, 1.00),
    (2, 6, 10, 20, 0.008, 0.040, 1200, 2200, 0.75, 0.95, 0.75, 0.95),
    (2, 6, 12, 25, 0.005, 0.030, 1500, 2500, 0.80, 1.00, 0.80, 1.00),
    (2, 6, 15, 30, 0.005, 0.020, 1800, 2500, 0.85, 1.00, 0.85, 1.00),
    (3, 6, 10, 30, 0.005, 0.030, 1500, 2500, 0.80, 1.00, 0.80, 1.00),
    (5, 8, 1, 5, 0.050, 0.200, 150, 800, 0.55, 0.75, 0.55, 0.75),
    (5, 8, 1, 5, 0.080, 0.300, 150, 600, 0.55, 0.75, 0.55, 0.75),
    (5, 8, 1, 8, 0.050, 0.150, 300, 900, 0.55, 0.80, 0.55, 0.80),
    (6, 8, 1, 5, 0.050, 0.200, 150, 800, 0.55, 0.70, 0.55, 0.70),
    (6, 8, 1, 5, 0.080, 0.300, 150, 600, 0.55, 0.70, 0.55, 0.70),
    (6, 8, 1, 8, 0.050, 0.150, 300, 900, 0.55, 0.75, 0.55, 0.75),
    (4, 8, 1, 5, 0.050, 0.200, 150, 1000, 0.55, 0.80, 0.55, 0.80),
    (4, 8, 1, 8, 0.050, 0.150, 300, 1000, 0.55, 0.80, 0.55, 0.80),
    (5, 7, 1, 3, 0.080, 0.250, 150, 500, 0.60, 0.80, 0.60, 0.80),
    (5, 7, 1, 3, 0.100, 0.300, 150, 400, 0.60, 0.80, 0.60, 0.80),
    (6, 8, 1, 3, 0.080, 0.250, 150, 500, 0.55, 0.75, 0.55, 0.75),
    (6, 8, 1, 3, 0.100, 0.300, 150, 400, 0.55, 0.75, 0.55, 0.75),
    (5, 8, 3, 8, 0.050, 0.120, 500, 1200, 0.55, 0.75, 0.55, 0.75),
    (5, 8, 1, 5, 0.030, 0.100, 600, 1500, 0.55, 0.75, 0.55, 0.75),
    (6, 8, 3, 8, 0.030, 0.100, 600, 1500, 0.55, 0.70, 0.55, 0.70),
    (4, 7, 1, 5, 0.050, 0.120, 500, 1200, 0.60, 0.80, 0.60, 0.80),
    (4, 7, 1, 8, 0.030, 0.100, 600, 1500, 0.60, 0.80, 0.60, 0.80),
    (5, 8, 1, 10, 0.050, 0.150, 300, 1200, 0.55, 0.75, 0.55, 0.75),
    (6, 8, 1, 10, 0.050, 0.150, 300, 1200, 0.55, 0.70, 0.55, 0.70),
    (4, 8, 1, 10, 0.050, 0.150, 300, 1200, 0.55, 0.80, 0.55, 0.80),
    (5, 8, 1, 6, 0.120, 0.300, 150, 350, 0.65, 0.90, 0.65, 0.90),
    (6, 8, 1, 6, 0.120, 0.300, 150, 350, 0.65, 0.90, 0.65, 0.90),
    (5, 8, 1, 6, 0.080, 0.200, 200, 600, 0.70, 1.00, 0.70, 1.00),
    (6, 8, 1, 6, 0.080, 0.200, 200, 600, 0.70, 1.00, 0.70, 1.00),
    (5, 7, 1, 4, 0.150, 0.300, 150, 300, 0.75, 1.00, 0.75, 1.00),
    (6, 8, 1, 4, 0.150, 0.300, 150, 300, 0.75, 1.00, 0.75, 1.00),
    (5, 8, 3, 10, 0.030, 0.120, 700, 1800, 0.55, 0.70, 0.55, 0.70),
    (6, 8, 3, 10, 0.030, 0.120, 700, 1800, 0.55, 0.65, 0.55, 0.65),
    (4, 8, 1, 6, 0.020, 0.080, 1000, 2200, 0.55, 0.75, 0.55, 0.75),
    (5, 8, 1, 6, 0.020, 0.080, 1000, 2500, 0.55, 0.70, 0.55, 0.70),
    (2, 4, 1, 8, 0.005, 0.020, 1500, 2500, 0.75, 1.00, 0.75, 1.00),
    (3, 5, 1, 8, 0.005, 0.020, 1500, 2500, 0.70, 0.95, 0.70, 0.95),
    (4, 6, 3, 10, 0.005, 0.020, 1800, 2500, 0.65, 0.90, 0.65, 0.90),
    (5, 7, 6, 15, 0.005, 0.020, 1800, 2500, 0.60, 0.85, 0.60, 0.85),
    (6, 8, 8, 20, 0.005, 0.020, 1800, 2500, 0.55, 0.80, 0.55, 0.80),
    (2, 4, 1, 8, 0.020, 0.050, 800, 1800, 0.75, 1.00, 0.75, 1.00),
    (3, 5, 1, 8, 0.020, 0.050, 800, 1800, 0.70, 0.95, 0.70, 0.95),
    (4, 6, 3, 10, 0.020, 0.050, 900, 1800, 0.65, 0.90, 0.65, 0.90),
    (5, 7, 6, 15, 0.020, 0.050, 1000, 2000, 0.60, 0.85, 0.60, 0.85),
    (6, 8, 8, 20, 0.020, 0.050, 1200, 2200, 0.55, 0.80, 0.55, 0.80),
    (2, 4, 1, 6, 0.080, 0.150, 150, 500, 0.80, 1.00, 0.80, 1.00),
    (3, 5, 1, 6, 0.080, 0.150, 150, 500, 0.75, 0.95, 0.75, 0.95),
    (4, 6, 1, 6, 0.080, 0.150, 150, 500, 0.70, 0.90, 0.70, 0.90),
    (5, 7, 3, 10, 0.080, 0.150, 200, 700, 0.65, 0.85, 0.65, 0.85),
    (6, 8, 6, 15, 0.080, 0.150, 250, 800, 0.60, 0.80, 0.60, 0.80),
    (2, 4, 8, 20, 0.020, 0.080, 500, 1400, 0.55, 0.70, 0.55, 0.70),
    (3, 5, 8, 20, 0.020, 0.080, 500, 1400, 0.55, 0.70, 0.55, 0.70),
    (4, 6, 10, 25, 0.020, 0.080, 700, 1600, 0.55, 0.70, 0.55, 0.70),
    (5, 7, 12, 30, 0.020, 0.080, 900, 1800, 0.55, 0.70, 0.55, 0.70),
    (6, 8, 12, 30, 0.020, 0.080, 1000, 2000, 0.55, 0.70, 0.55, 0.70),
]

assert len(_PRESET_3V_ORDERED) == 100, f"Expected 100 ordered presets for 3v1-v100, got {len(_PRESET_3V_ORDERED)}"

for idx, (md_lo, md_hi, mcw_lo, mcw_hi, lr_lo, lr_hi, ne_lo, ne_hi, ss_lo, ss_hi, cs_lo, cs_hi) in enumerate(_PRESET_3V_ORDERED, start=1):
    SEARCH_SPACE_PRESETS[f"3v{idx}"] = {
        "max_depth": int_range(md_lo, md_hi),
        "min_child_weight": int_range(mcw_lo, mcw_hi),
        "learning_rate": float_range(lr_lo, lr_hi),
        "n_estimators": int_range(ne_lo, ne_hi),
        "subsample": float_range(ss_lo, ss_hi),
        "colsample_bytree": float_range(cs_lo, cs_hi),
    }

if legacy_search_space_tag:
    print(f"Warning: legacy search space tag '{legacy_search_space_tag}' is ignored in single-space mode.")
if OPTUNA_OBJECTIVE_INDEX != 1:
    raise ValueError(
        f"Unsupported optuna objective index '{OPTUNA_OBJECTIVE_INDEX}' in tag '{train_tag}'. "
        "Current workflow supports objective 1 (validation AUC) only."
    )
OPTUNA_SEARCH_SPACE = OPTUNA_SPACES[DATASET_TOKEN][CHANNEL][OPTUNA_SPACE_VERSION]
OPTUNA_TRAINING_CFG = OPTUNA_TRAINING_OPTIONS[DATASET_TOKEN][CHANNEL][OPTUNA_SPACE_VERSION]
EARLY_STOPPING_ROUNDS = OPTUNA_TRAINING_CFG.get("early_stopping_rounds")


def suggest_param(trial, name, config):
    if config["type"] == "int":
        return trial.suggest_int(name, config["low"], config["high"])
    if config["type"] == "float":
        return trial.suggest_float(name, config["low"], config["high"], log=config.get("log", False))
    raise ValueError(f"Unsupported parameter type for {name}: {config['type']}")


def save_feature_importance(model, feature_names, train_tag):
    importance_pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    print("Feature importance ranking:")
    for rank, (name, score) in enumerate(importance_pairs, start=1):
        print(f"  {rank}. {name}: {score:.6f}")

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
    print(f"Feature importance saved to: {importance_json_path}")

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
    print(f"Feature importance plot saved to: {cumulative_plot_path}")


def build_roc_payload(y_true, score):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    return {
        "auc": float(roc_auc),
        "threshold": [float(x) for x in thresholds],
        "tpr": [float(x) for x in tpr],
        "fpr": [float(x) for x in fpr],
        "background_rejection": [float(1.0 - x) for x in fpr],
        "signal_efficiency": [float(x) for x in tpr],
    }


def summarize_top_trials(study, search_space, top_n=20):
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    ranked_trials = sorted(completed_trials, key=lambda trial: trial.value, reverse=True)
    top_trials = ranked_trials[:top_n]

    range_summary = {}
    for param_name, config in search_space.items():
        values = [trial.params[param_name] for trial in top_trials if param_name in trial.params]
        if not values:
            continue
        top_min = min(values)
        top_max = max(values)
        initial_low = config.get("low")
        initial_high = config.get("high")
        if config.get("type") == "int":
            top_min = int(round(top_min))
            top_max = int(round(top_max))
        else:
            top_min = float(top_min)
            top_max = float(top_max)
        range_summary[param_name] = {
            "parameter_type": config.get("type"),
            "initial_range_low": initial_low,
            "initial_range_high": initial_high,
            "top20_range_low": top_min,
            "top20_range_high": top_max,
            "n_top20_unique_values": len(set(values)),
            "comparison_to_initial": {
                "low_shift": float(top_min) - float(initial_low),
                "high_shift": float(top_max) - float(initial_high),
                "is_narrowed_or_equal": float(top_min) >= float(initial_low) and float(top_max) <= float(initial_high),
            },
        }
        if "log" in config:
            range_summary[param_name]["sampling_log"] = bool(config["log"])

    return top_trials, range_summary


def save_top20_range_json(train_tag, search_space_tag, search_space, study, top_n=20):
    top_trials, range_summary = summarize_top_trials(study, search_space, top_n=top_n)
    output_path = os.path.join(condor_model_dir(train_tag), "optuna_top20_ranges.json")
    payload = {
        "field_meanings": {
            "ranking_metric": "Optuna objective value (validation AUC) sorted descending",
            "initial_range_low/high": "The low/high boundary of this run's search space at start",
            "top20_range_low/high": "Min/max of the parameter among the best 20 completed trials",
            "low_shift/high_shift": "top20 boundary minus initial boundary",
            "is_narrowed_or_equal": "Whether top20 range is inside or equal to initial range",
        },
        "train_tag": train_tag,
        "search_space_tag": search_space_tag,
        "dataset_source": DATASET_SOURCE,
        "dataset_year": DATASET_YEAR,
        "selection_profile": SELECTION_PROFILE,
        "n_trials_requested": number_trials,
        "n_trials_completed": len([trial for trial in study.trials if trial.value is not None]),
        "n_top_trials_used": min(top_n, len([trial for trial in study.trials if trial.value is not None])),
        "ranking_metric": "validation_auc",
        "optimization_direction": "maximize",
        "initial_search_space": search_space,
        "parameter_range_summary": range_summary,
        "top_trials": [
            {
                "rank": rank,
                "trial_number": int(trial.number),
                "objective_value": float(trial.value),
                "params": {
                    key: (float(value) if isinstance(value, float) else value)
                    for key, value in trial.params.items()
                },
            }
            for rank, trial in enumerate(top_trials, start=1)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Top20 range summary saved to: {output_path}")
    return output_path


ensure_dir(condor_model_dir(train_tag))
ensure_dir(condor_training_dir(train_tag))

print(f"Using feature set: {feature_set_tag}")
print(f"Input columns: {input_columns}")
print("Using single optuna search space.")
print(f"Optuna objective index: {OPTUNA_OBJECTIVE_INDEX}")
print(f"Optuna space version: {OPTUNA_SPACE_VERSION}")
print(f"Optuna early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")
print(f"Dataset source: {DATASET_SOURCE}")
print(f"Selection profile: {SELECTION_PROFILE}")
print(json.dumps(OPTUNA_SEARCH_SPACE, indent=2))

ak_sig = uproot.concatenate(SIG_PATH, library="pd")
ak_bkg = uproot.concatenate(BKG_PATH, library="pd")

ak_sig = apply_selection(ak_sig, SIGNAL_SELECTION, "signal_selection")
ak_bkg = apply_selection(ak_bkg, BACKGROUND_SELECTION, "background_selection")

ak_sig["is_sig"] = True
ak_bkg["is_sig"] = False
ak_sig["is_bkg"] = False
ak_bkg["is_bkg"] = True

df_raw = pd.concat([ak_sig, ak_bkg], axis=0, ignore_index=True)

scaler = StandardScaler()
df_trans = pd.DataFrame(
    scaler.fit_transform(df_raw[input_columns]),
    columns=[f"{col}_trans" for col in input_columns],
    index=df_raw.index,
)
df = pd.concat([df_trans, df_raw], axis=1)

trans_columns = [f"{col}_trans" for col in input_columns]
X = df[trans_columns]
y = df[["is_sig", "is_bkg"]]

X_train, X_valtest, y_train, y_valtest = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y["is_sig"],
    random_state=FIXED_MODEL_PARAMS["random_state"],
)
X_val, X_test, y_val, y_test = train_test_split(
    X_valtest,
    y_valtest,
    test_size=0.5,
    stratify=y_valtest["is_sig"],
    random_state=FIXED_MODEL_PARAMS["random_state"],
)

n_sig = (y_train["is_sig"] == 1).sum()
n_bkg = (y_train["is_sig"] == 0).sum()
pos_weight = n_bkg / n_sig


def objective(trial):
    params = {
        "booster": FIXED_MODEL_PARAMS["booster"],
        "objective": FIXED_MODEL_PARAMS["objective"],
        "eval_metric": FIXED_MODEL_PARAMS["eval_metric"],
        "random_state": FIXED_MODEL_PARAMS["random_state"],
        "scale_pos_weight": pos_weight,
        "n_jobs": 4,
    }
    if EARLY_STOPPING_ROUNDS is not None:
        params["early_stopping_rounds"] = int(EARLY_STOPPING_ROUNDS)
    for name, config in OPTUNA_SEARCH_SPACE.items():
        params[name] = suggest_param(trial, name, config)

    model = XGBClassifier(**params)
    fit_kwargs = {}
    if EARLY_STOPPING_ROUNDS is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val["is_sig"])]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train["is_sig"], **fit_kwargs)

    pred = model.predict_proba(X_val)[:, 1]
    val_fpr, val_tpr, _ = roc_curve(y_val["is_sig"].astype(int).to_numpy(), pred)
    val_auc = auc(val_fpr, val_tpr)
    trial.set_user_attr("validation_auc", float(val_auc))
    return val_auc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=number_trials)

print("Best params:", study.best_params)
print(f"Best validation AUC: {study.best_value:.6f}")
top20_ranges_json_path = save_top20_range_json(
    train_tag=train_tag,
    search_space_tag=OPTUNA_SPACE_VERSION,
    search_space=OPTUNA_SEARCH_SPACE,
    study=study,
    top_n=20,
)

xgbc = XGBClassifier(
    **study.best_params,
    booster=FIXED_MODEL_PARAMS["booster"],
    objective=FIXED_MODEL_PARAMS["objective"],
    eval_metric=FIXED_MODEL_PARAMS["eval_metric"],
    random_state=FIXED_MODEL_PARAMS["random_state"],
    scale_pos_weight=pos_weight,
    n_jobs=4,
    **({"early_stopping_rounds": int(EARLY_STOPPING_ROUNDS)} if EARLY_STOPPING_ROUNDS is not None else {}),
)
final_fit_kwargs = {}
if EARLY_STOPPING_ROUNDS is not None:
    final_fit_kwargs["eval_set"] = [(X_val, y_val["is_sig"])]
    final_fit_kwargs["verbose"] = False
xgbc.fit(X_train, y_train["is_sig"], **final_fit_kwargs)

train_score_xgb = xgbc.predict_proba(X_train)
test_score_xgb = xgbc.predict_proba(X_test)
train_score_xgb_all = train_score_xgb[:, 1]
test_score_xgb_sig = test_score_xgb[y_test["is_sig"]][:, 1]
test_score_xgb_bkg = test_score_xgb[y_test["is_bkg"]][:, 1]
test_score_xgb_all = test_score_xgb[:, 1]
train_y_true = y_train["is_sig"].astype(int).to_numpy()
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
print(f"Score plot saved to: {score_plot_path}")

train_roc = build_roc_payload(train_y_true, train_score_xgb_all)
test_roc = build_roc_payload(test_y_true, test_score_xgb_all)

roc_json_path = os.path.join(condor_training_dir(train_tag), "roc.json")
with open(roc_json_path, "w") as f:
    json.dump(
        {
            "mode": "optuna_xgboost",
            "train": train_roc,
            "test": test_roc,
        },
        f,
        indent=2,
    )
print(f"ROC saved to: {roc_json_path}")

roc_plot_path = os.path.join(condor_training_dir(train_tag), "roc.pdf")
plt.figure(figsize=(6, 6))
plt.plot(
    train_roc["signal_efficiency"],
    train_roc["background_rejection"],
    linewidth=2,
    label=f"Train AUC = {train_roc['auc']:.4f}",
)
plt.plot(
    test_roc["signal_efficiency"],
    test_roc["background_rejection"],
    linewidth=2,
    label=f"Test AUC = {test_roc['auc']:.4f}",
)
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.title("ROC Curve")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(roc_plot_path)
plt.close()
print(f"ROC plot saved to: {roc_plot_path}")

trained_model_path = condor_model_path(train_tag)
joblib.dump(xgbc, trained_model_path)
print(f"Model saved to: {trained_model_path}")

trained_scaler_path = condor_scaler_path(train_tag)
joblib.dump(scaler, trained_scaler_path)
print(f"Scaler saved to: {trained_scaler_path}")

config = {
    "input_columns": input_columns,
    "trans_columns": trans_columns,
}
config_output_path = condor_model_config_path(train_tag)
with open(config_output_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_output_path}")

save_feature_importance(xgbc, input_columns, train_tag)

save_run_metadata(
    train_tag=train_tag,
    training_script="condor_optuna_XGBoost.py",
    signal_path=SIG_PATH,
    background_path=BKG_PATH,
    signal_selection=SIGNAL_SELECTION,
    background_selection=BACKGROUND_SELECTION,
    input_columns=input_columns,
    trans_columns=trans_columns,
    pos_weight=pos_weight,
    fixed_model_params={
        **FIXED_MODEL_PARAMS,
        "scale_pos_weight": float(pos_weight),
        "n_jobs": 4,
    },
    best_model_params={
        **{key: (float(value) if isinstance(value, float) else value) for key, value in study.best_params.items()},
        **FIXED_MODEL_PARAMS,
        "scale_pos_weight": float(pos_weight),
        "n_jobs": 4,
    },
    is_optuna=True,
    optuna_n_trials=number_trials,
    optimized_hyperparameters=list(OPTUNA_SEARCH_SPACE.keys()),
    hyperparameter_search_space=OPTUNA_SEARCH_SPACE,
    optimization_metric="max validation AUC",
    best_objective_value=study.best_value,
    notes={
        "search_space_tag": "single",
        "optuna_objective_index": OPTUNA_OBJECTIVE_INDEX,
        "optuna_space_version": OPTUNA_SPACE_VERSION,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_validation_auc": float(study.best_trial.user_attrs.get("validation_auc", study.best_value)),
        "train_auc": float(train_roc["auc"]),
        "test_auc": float(test_roc["auc"]),
        "roc_json_path": roc_json_path,
        "roc_plot_path": roc_plot_path,
        "dataset_source": DATASET_SOURCE,
        "dataset_year": DATASET_YEAR,
        "selection_profile": SELECTION_PROFILE,
        "optuna_top20_ranges_json_path": top20_ranges_json_path,
    },
)

print("Training complete. Model artifacts saved.")
