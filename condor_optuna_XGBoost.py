import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils.paths import (
    ensure_dir,
    feature_importance_cumulative_path,
    feature_importance_path,
    model_config_path,
    model_dir,
    model_path,
    scaler_path,
    training_dir,
    training_score_path,
)
from utils.run_metadata import save_run_metadata

if len(sys.argv) != 3:
    print("Usage: python3 condor_optuna_XGBoost.py <train_tag> <search_space_tag>")
    sys.exit(1)

train_tag = sys.argv[1]
search_space_tag = sys.argv[2]
number_trials = int(os.environ.get("OPTUNA_N_TRIALS", "100"))

SIG_PATH = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC.root:ntmix"
BKG_PATH = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA0.root:ntmix"

input_columns = ["Btrk1dR", "Btrk2Pt", "BtrkPtimb", "Bchi2Prob"]
SIGNAL_SELECTION = "isX3872 == 1"
BACKGROUND_SELECTION = "(3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)"
FIXED_MODEL_PARAMS = {
    "eval_metric": "logloss",
    "random_state": 42,
}

SEARCH_SPACE_PRESETS = {
    "v11": {
        "n_estimators": {"type": "int", "low": 500, "high": 1600},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.08, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 4},
        "min_child_weight": {"type": "int", "low": 6, "high": 16},
        "subsample": {"type": "float", "low": 0.65, "high": 0.85},
        "colsample_bytree": {"type": "float", "low": 0.65, "high": 0.85},
        "gamma": {"type": "float", "low": 1.0, "high": 4.0},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_lambda": {"type": "float", "low": 3.0, "high": 10.0},
    },
    "v12": {
        "n_estimators": {"type": "int", "low": 700, "high": 2000},
        "learning_rate": {"type": "float", "low": 0.015, "high": 0.06, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 4},
        "min_child_weight": {"type": "int", "low": 4, "high": 12},
        "subsample": {"type": "float", "low": 0.70, "high": 0.90},
        "colsample_bytree": {"type": "float", "low": 0.70, "high": 0.90},
        "gamma": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_alpha": {"type": "float", "low": 0.2, "high": 2.0},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 8.0},
    },
    "v13": {
        "n_estimators": {"type": "int", "low": 350, "high": 1100},
        "learning_rate": {"type": "float", "low": 0.05, "high": 0.12},
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "min_child_weight": {"type": "int", "low": 3, "high": 10},
        "subsample": {"type": "float", "low": 0.70, "high": 0.90},
        "colsample_bytree": {"type": "float", "low": 0.70, "high": 0.90},
        "gamma": {"type": "float", "low": 0.5, "high": 2.0},
        "reg_alpha": {"type": "float", "low": 0.1, "high": 1.5},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 6.0},
    },
    "v14": {
        "n_estimators": {"type": "int", "low": 350, "high": 1200},
        "learning_rate": {"type": "float", "low": 0.04, "high": 0.10},
        "max_depth": {"type": "int", "low": 4, "high": 5},
        "min_child_weight": {"type": "int", "low": 6, "high": 18},
        "subsample": {"type": "float", "low": 0.75, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 0.95},
        "gamma": {"type": "float", "low": 1.0, "high": 4.0},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_lambda": {"type": "float", "low": 3.0, "high": 10.0},
    },
    "v15": {
        "n_estimators": {"type": "int", "low": 600, "high": 1800},
        "learning_rate": {"type": "float", "low": 0.015, "high": 0.05, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 4},
        "min_child_weight": {"type": "int", "low": 8, "high": 20},
        "subsample": {"type": "float", "low": 0.60, "high": 0.80},
        "colsample_bytree": {"type": "float", "low": 0.60, "high": 0.80},
        "gamma": {"type": "float", "low": 1.0, "high": 5.0},
        "reg_alpha": {"type": "float", "low": 1.0, "high": 4.0},
        "reg_lambda": {"type": "float", "low": 4.0, "high": 12.0},
    },
    "v16": {
        "n_estimators": {"type": "int", "low": 450, "high": 1300},
        "learning_rate": {"type": "float", "low": 0.03, "high": 0.09},
        "max_depth": {"type": "int", "low": 3, "high": 4},
        "min_child_weight": {"type": "int", "low": 4, "high": 10},
        "subsample": {"type": "float", "low": 0.85, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.85, "high": 1.0},
        "gamma": {"type": "float", "low": 0.2, "high": 1.5},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 6.0},
    },
    "v17": {
        "n_estimators": {"type": "int", "low": 900, "high": 2400},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "min_child_weight": {"type": "int", "low": 3, "high": 8},
        "subsample": {"type": "float", "low": 0.70, "high": 0.90},
        "colsample_bytree": {"type": "float", "low": 0.70, "high": 0.90},
        "gamma": {"type": "float", "low": 0.3, "high": 2.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 8.0},
    },
    "v18": {
        "n_estimators": {"type": "int", "low": 250, "high": 900},
        "learning_rate": {"type": "float", "low": 0.08, "high": 0.16},
        "max_depth": {"type": "int", "low": 4, "high": 6},
        "min_child_weight": {"type": "int", "low": 2, "high": 6},
        "subsample": {"type": "float", "low": 0.75, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 0.95},
        "gamma": {"type": "float", "low": 0.0, "high": 1.2},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 4.0},
    },
    "v19": {
        "n_estimators": {"type": "int", "low": 400, "high": 1300},
        "learning_rate": {"type": "float", "low": 0.03, "high": 0.10},
        "max_depth": {"type": "int", "low": 4, "high": 6},
        "min_child_weight": {"type": "int", "low": 8, "high": 22},
        "subsample": {"type": "float", "low": 0.75, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 0.95},
        "gamma": {"type": "float", "low": 1.5, "high": 5.0},
        "reg_alpha": {"type": "float", "low": 1.0, "high": 4.0},
        "reg_lambda": {"type": "float", "low": 4.0, "high": 12.0},
    },
    "v20": {
        "n_estimators": {"type": "int", "low": 250, "high": 800},
        "learning_rate": {"type": "float", "low": 0.10, "high": 0.18},
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "min_child_weight": {"type": "int", "low": 3, "high": 8},
        "subsample": {"type": "float", "low": 0.80, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.80, "high": 1.0},
        "gamma": {"type": "float", "low": 0.2, "high": 2.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 6.0},
    },
    "v21": {
        "n_estimators": {"type": "int", "low": 600, "high": 1800},
        "learning_rate": {"type": "float", "low": 0.015, "high": 0.05, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 3},
        "min_child_weight": {"type": "int", "low": 10, "high": 24},
        "subsample": {"type": "float", "low": 0.55, "high": 0.75},
        "colsample_bytree": {"type": "float", "low": 0.55, "high": 0.75},
        "gamma": {"type": "float", "low": 2.0, "high": 6.0},
        "reg_alpha": {"type": "float", "low": 1.5, "high": 5.0},
        "reg_lambda": {"type": "float", "low": 5.0, "high": 14.0},
    },
    "v22": {
        "n_estimators": {"type": "int", "low": 900, "high": 2600},
        "learning_rate": {"type": "float", "low": 0.008, "high": 0.03, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 4},
        "min_child_weight": {"type": "int", "low": 6, "high": 14},
        "subsample": {"type": "float", "low": 0.65, "high": 0.85},
        "colsample_bytree": {"type": "float", "low": 0.65, "high": 0.85},
        "gamma": {"type": "float", "low": 0.8, "high": 3.5},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 2.5},
        "reg_lambda": {"type": "float", "low": 3.0, "high": 9.0},
    },
    "v23": {
        "n_estimators": {"type": "int", "low": 350, "high": 1100},
        "learning_rate": {"type": "float", "low": 0.05, "high": 0.12},
        "max_depth": {"type": "int", "low": 3, "high": 4},
        "min_child_weight": {"type": "int", "low": 4, "high": 12},
        "subsample": {"type": "float", "low": 0.60, "high": 0.80},
        "colsample_bytree": {"type": "float", "low": 0.60, "high": 0.80},
        "gamma": {"type": "float", "low": 0.5, "high": 2.5},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 2.0},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 7.0},
    },
    "v24": {
        "n_estimators": {"type": "int", "low": 300, "high": 900},
        "learning_rate": {"type": "float", "low": 0.08, "high": 0.18},
        "max_depth": {"type": "int", "low": 5, "high": 7},
        "min_child_weight": {"type": "int", "low": 1, "high": 4},
        "subsample": {"type": "float", "low": 0.80, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.80, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 4.0},
    },
    "v25": {
        "n_estimators": {"type": "int", "low": 500, "high": 1600},
        "learning_rate": {"type": "float", "low": 0.03, "high": 0.08},
        "max_depth": {"type": "int", "low": 5, "high": 6},
        "min_child_weight": {"type": "int", "low": 1, "high": 5},
        "subsample": {"type": "float", "low": 0.65, "high": 0.85},
        "colsample_bytree": {"type": "float", "low": 0.65, "high": 0.85},
        "gamma": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.2},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 5.0},
    },
    "v26": {
        "n_estimators": {"type": "int", "low": 800, "high": 2200},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
        "max_depth": {"type": "int", "low": 4, "high": 6},
        "min_child_weight": {"type": "int", "low": 2, "high": 8},
        "subsample": {"type": "float", "low": 0.85, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.85, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 1.2},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 5.0},
    },
    "v27": {
        "n_estimators": {"type": "int", "low": 350, "high": 1200},
        "learning_rate": {"type": "float", "low": 0.04, "high": 0.12},
        "max_depth": {"type": "int", "low": 4, "high": 6},
        "min_child_weight": {"type": "int", "low": 8, "high": 18},
        "subsample": {"type": "float", "low": 0.70, "high": 0.90},
        "colsample_bytree": {"type": "float", "low": 0.70, "high": 0.90},
        "gamma": {"type": "float", "low": 1.5, "high": 4.5},
        "reg_alpha": {"type": "float", "low": 1.0, "high": 3.5},
        "reg_lambda": {"type": "float", "low": 4.0, "high": 12.0},
    },
    "v28": {
        "n_estimators": {"type": "int", "low": 250, "high": 850},
        "learning_rate": {"type": "float", "low": 0.12, "high": 0.22},
        "max_depth": {"type": "int", "low": 6, "high": 8},
        "min_child_weight": {"type": "int", "low": 1, "high": 3},
        "subsample": {"type": "float", "low": 0.75, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 0.95},
        "gamma": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 3.5},
    },
    "v29": {
        "n_estimators": {"type": "int", "low": 450, "high": 1400},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.07, "log": True},
        "max_depth": {"type": "int", "low": 5, "high": 7},
        "min_child_weight": {"type": "int", "low": 6, "high": 14},
        "subsample": {"type": "float", "low": 0.55, "high": 0.75},
        "colsample_bytree": {"type": "float", "low": 0.55, "high": 0.75},
        "gamma": {"type": "float", "low": 1.0, "high": 3.5},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 2.5},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 8.0},
    },
    "v30": {
        "n_estimators": {"type": "int", "low": 500, "high": 1700},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.06, "log": True},
        "max_depth": {"type": "int", "low": 4, "high": 5},
        "min_child_weight": {"type": "int", "low": 2, "high": 10},
        "subsample": {"type": "float", "low": 0.60, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.60, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 3.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 2.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 8.0},
    },
    "v31": {
        "n_estimators": {"type": "int", "low": 700, "high": 2200},
        "learning_rate": {"type": "float", "low": 0.006, "high": 0.02, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 3},
        "min_child_weight": {"type": "int", "low": 12, "high": 28},
        "subsample": {"type": "float", "low": 0.45, "high": 0.65},
        "colsample_bytree": {"type": "float", "low": 0.45, "high": 0.65},
        "gamma": {"type": "float", "low": 2.5, "high": 7.0},
        "reg_alpha": {"type": "float", "low": 2.0, "high": 6.0},
        "reg_lambda": {"type": "float", "low": 6.0, "high": 16.0},
    },
    "v32": {
        "n_estimators": {"type": "int", "low": 250, "high": 700},
        "learning_rate": {"type": "float", "low": 0.14, "high": 0.26},
        "max_depth": {"type": "int", "low": 2, "high": 3},
        "min_child_weight": {"type": "int", "low": 1, "high": 4},
        "subsample": {"type": "float", "low": 0.85, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.85, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.6},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 3.5},
    },
    "v33": {
        "n_estimators": {"type": "int", "low": 900, "high": 2600},
        "learning_rate": {"type": "float", "low": 0.008, "high": 0.03, "log": True},
        "max_depth": {"type": "int", "low": 5, "high": 7},
        "min_child_weight": {"type": "int", "low": 1, "high": 5},
        "subsample": {"type": "float", "low": 0.80, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.80, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 4.0},
    },
    "v34": {
        "n_estimators": {"type": "int", "low": 300, "high": 1000},
        "learning_rate": {"type": "float", "low": 0.06, "high": 0.16},
        "max_depth": {"type": "int", "low": 6, "high": 8},
        "min_child_weight": {"type": "int", "low": 10, "high": 24},
        "subsample": {"type": "float", "low": 0.70, "high": 0.90},
        "colsample_bytree": {"type": "float", "low": 0.70, "high": 0.90},
        "gamma": {"type": "float", "low": 2.0, "high": 6.0},
        "reg_alpha": {"type": "float", "low": 1.5, "high": 5.0},
        "reg_lambda": {"type": "float", "low": 5.0, "high": 14.0},
    },
    "v35": {
        "n_estimators": {"type": "int", "low": 450, "high": 1400},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.07, "log": True},
        "max_depth": {"type": "int", "low": 4, "high": 5},
        "min_child_weight": {"type": "int", "low": 2, "high": 8},
        "subsample": {"type": "float", "low": 0.40, "high": 0.60},
        "colsample_bytree": {"type": "float", "low": 0.40, "high": 0.60},
        "gamma": {"type": "float", "low": 0.5, "high": 2.5},
        "reg_alpha": {"type": "float", "low": 0.3, "high": 2.0},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 8.0},
    },
    "v36": {
        "n_estimators": {"type": "int", "low": 250, "high": 850},
        "learning_rate": {"type": "float", "low": 0.12, "high": 0.24},
        "max_depth": {"type": "int", "low": 5, "high": 7},
        "min_child_weight": {"type": "int", "low": 1, "high": 4},
        "subsample": {"type": "float", "low": 0.55, "high": 0.75},
        "colsample_bytree": {"type": "float", "low": 0.55, "high": 0.75},
        "gamma": {"type": "float", "low": 0.0, "high": 1.2},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 4.0},
    },
    "v37": {
        "n_estimators": {"type": "int", "low": 600, "high": 1800},
        "learning_rate": {"type": "float", "low": 0.015, "high": 0.05, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "min_child_weight": {"type": "int", "low": 1, "high": 6},
        "subsample": {"type": "float", "low": 0.90, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.35, "high": 0.55},
        "gamma": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 5.0},
    },
    "v38": {
        "n_estimators": {"type": "int", "low": 500, "high": 1600},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.08},
        "max_depth": {"type": "int", "low": 4, "high": 6},
        "min_child_weight": {"type": "int", "low": 12, "high": 30},
        "subsample": {"type": "float", "low": 0.85, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.85, "high": 1.0},
        "gamma": {"type": "float", "low": 2.0, "high": 6.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.5},
        "reg_lambda": {"type": "float", "low": 6.0, "high": 18.0},
    },
    "v39": {
        "n_estimators": {"type": "int", "low": 350, "high": 1100},
        "learning_rate": {"type": "float", "low": 0.03, "high": 0.09},
        "max_depth": {"type": "int", "low": 6, "high": 8},
        "min_child_weight": {"type": "int", "low": 3, "high": 8},
        "subsample": {"type": "float", "low": 0.40, "high": 0.65},
        "colsample_bytree": {"type": "float", "low": 0.80, "high": 1.0},
        "gamma": {"type": "float", "low": 0.3, "high": 2.0},
        "reg_alpha": {"type": "float", "low": 0.2, "high": 1.5},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 6.0},
    },
    "v40": {
        "n_estimators": {"type": "int", "low": 250, "high": 1800},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 7},
        "min_child_weight": {"type": "int", "low": 1, "high": 20},
        "subsample": {"type": "float", "low": 0.40, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.35, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 6.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 5.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 16.0},
    },
}

if search_space_tag not in SEARCH_SPACE_PRESETS:
    print(f"Unknown search_space_tag: {search_space_tag}")
    print(f"Available presets: {sorted(SEARCH_SPACE_PRESETS)}")
    sys.exit(1)

OPTUNA_SEARCH_SPACE = SEARCH_SPACE_PRESETS[search_space_tag]


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

    importance_json_path = feature_importance_path(train_tag)
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

    cumulative_plot_path = feature_importance_cumulative_path(train_tag)
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


ensure_dir(model_dir(train_tag))
ensure_dir(training_dir(train_tag))

print(f"Using search space preset: {search_space_tag}")
print(json.dumps(OPTUNA_SEARCH_SPACE, indent=2))

ak_sig = uproot.concatenate(SIG_PATH, library="pd")
ak_bkg = uproot.concatenate(BKG_PATH, library="pd")

ak_sig = ak_sig[ak_sig["isX3872"] == 1]
ak_bkg = ak_bkg[
    ((ak_bkg["Bmass"] > 3.75) & (ak_bkg["Bmass"] < 3.83))
    | ((ak_bkg["Bmass"] > 3.91) & (ak_bkg["Bmass"] < 4.00))
]

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


def best_significance_from_scores(signal_scores, background_scores):
    if len(signal_scores) == 0 or len(background_scores) == 0:
        return 0.0, 0.0, 0, 0

    candidate_cuts = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.999, 400),
                np.quantile(signal_scores, np.linspace(0.05, 0.95, 19)),
                np.quantile(background_scores, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )

    best_significance = 0.0
    best_cut = 0.0
    best_signal = 0
    best_background = 0

    for cut in candidate_cuts:
        passed_signal = int(np.count_nonzero(signal_scores > cut))
        passed_background = int(np.count_nonzero(background_scores > cut))
        total = passed_signal + passed_background
        if total <= 0:
            continue

        significance = passed_signal / np.sqrt(total)
        if significance > best_significance:
            best_significance = significance
            best_cut = float(cut)
            best_signal = passed_signal
            best_background = passed_background

    return best_significance, best_cut, best_signal, best_background


def objective(trial):
    params = {
        "eval_metric": FIXED_MODEL_PARAMS["eval_metric"],
        "random_state": FIXED_MODEL_PARAMS["random_state"],
        "scale_pos_weight": pos_weight,
        "n_jobs": 4,
    }
    for name, config in OPTUNA_SEARCH_SPACE.items():
        params[name] = suggest_param(trial, name, config)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train["is_sig"])

    pred = model.predict_proba(X_val)[:, 1]
    sig = pred[y_val["is_sig"] == 1]
    bkg = pred[y_val["is_sig"] == 0]
    best_significance, best_cut, best_signal, best_background = best_significance_from_scores(sig, bkg)
    trial.set_user_attr("best_cut", best_cut)
    trial.set_user_attr("best_signal_yield", best_signal)
    trial.set_user_attr("best_background_yield", best_background)
    return best_significance


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=number_trials)

print("Best params:", study.best_params)
print(f"Best validation cut: {study.best_trial.user_attrs.get('best_cut', 0.0):.4f}")
print(f"Best validation S: {study.best_trial.user_attrs.get('best_signal_yield', 0)}")
print(f"Best validation B: {study.best_trial.user_attrs.get('best_background_yield', 0)}")
print(f"Best validation S/sqrt(S+B): {study.best_value:.6f}")

xgbc = XGBClassifier(
    **study.best_params,
    eval_metric=FIXED_MODEL_PARAMS["eval_metric"],
    random_state=FIXED_MODEL_PARAMS["random_state"],
    scale_pos_weight=pos_weight,
    n_jobs=4,
)
xgbc.fit(X_train, y_train["is_sig"])

test_score_xgb = xgbc.predict_proba(X_test)
test_score_xgb_sig = test_score_xgb[y_test["is_sig"]][:, 1]
test_score_xgb_bkg = test_score_xgb[y_test["is_bkg"]][:, 1]

score_plot_path = training_score_path(train_tag)
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

trained_model_path = model_path(train_tag)
joblib.dump(xgbc, trained_model_path)
print(f"Model saved to: {trained_model_path}")

trained_scaler_path = scaler_path(train_tag)
joblib.dump(scaler, trained_scaler_path)
print(f"Scaler saved to: {trained_scaler_path}")

config = {
    "input_columns": input_columns,
    "trans_columns": trans_columns,
}
config_output_path = model_config_path(train_tag)
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
    optimization_metric="max validation S/sqrt(S+B) from score-cut scan",
    best_objective_value=study.best_value,
    notes={
        "search_space_tag": search_space_tag,
        "best_validation_cut": float(study.best_trial.user_attrs.get("best_cut", 0.0)),
        "best_validation_signal_yield": int(study.best_trial.user_attrs.get("best_signal_yield", 0)),
        "best_validation_background_yield": int(study.best_trial.user_attrs.get("best_background_yield", 0)),
    },
)

print("Training complete. Model artifacts saved.")
