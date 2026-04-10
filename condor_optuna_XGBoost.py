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

if len(sys.argv) != 3:
    print("Usage: python3 condor_optuna_XGBoost.py <train_tag> <search_space_tag>")
    sys.exit(1)

train_tag = sys.argv[1]
search_space_tag = sys.argv[2]
number_trials = int(os.environ.get("OPTUNA_N_TRIALS", "100"))

SIG_PATH = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC.root:ntmix"
BKG_PATH = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA0.root:ntmix"

FEATURE_SETS = {
    "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
    "5v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt"],
    "7v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Balpha", "Bnorm_trk1Dxy"],
}

feature_set_tag = None
for candidate in FEATURE_SETS:
    if f"_{candidate}_" in train_tag:
        feature_set_tag = candidate
        break

if feature_set_tag is None:
    print(f"Unable to infer feature set from train_tag: {train_tag}")
    print(f"Supported feature sets: {sorted(FEATURE_SETS)}")
    sys.exit(1)

input_columns = FEATURE_SETS[feature_set_tag]
SIGNAL_SELECTION = "isX3872 == 1 and abs(By) < 1.6 and 15 < Bpt < 50 and 0 < CentBin < 90"
BACKGROUND_SELECTION = "((3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)) and abs(By) < 1.6 and 15 < Bpt < 50 and 0 < CentBin < 90"
SIGNAL_SCALE_FACTOR = 3491.0 / 70439.0
SIGNAL_WINDOW_WIDTH = 3.91 - 3.83
SIDEBAND_WINDOW_WIDTH = (3.83 - 3.75) + (4.00 - 3.91)
FULL_TO_PARTIAL_DATA_SCALE = 31762286.0 / 994663.0
BACKGROUND_SCALE_FACTOR = (SIGNAL_WINDOW_WIDTH / SIDEBAND_WINDOW_WIDTH) * FULL_TO_PARTIAL_DATA_SCALE
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


ensure_dir(condor_model_dir(train_tag))
ensure_dir(condor_training_dir(train_tag))

print(f"Using feature set: {feature_set_tag}")
print(f"Input columns: {input_columns}")
print(f"Using search space preset: {search_space_tag}")
print(json.dumps(OPTUNA_SEARCH_SPACE, indent=2))

ak_sig = uproot.concatenate(SIG_PATH, library="pd")
ak_bkg = uproot.concatenate(BKG_PATH, library="pd")

ak_sig = ak_sig[
    (ak_sig["isX3872"] == 1)
    & (np.abs(ak_sig["By"]) < 1.6)
    & (ak_sig["Bpt"] > 15.0)
    & (ak_sig["Bpt"] < 50.0)
    & (ak_sig["CentBin"] > 0)
    & (ak_sig["CentBin"] < 90)
]
ak_bkg = ak_bkg[
    ((ak_bkg["Bmass"] > 3.75) & (ak_bkg["Bmass"] < 3.83))
    | ((ak_bkg["Bmass"] > 3.91) & (ak_bkg["Bmass"] < 4.00))
]
ak_bkg = ak_bkg[
    (np.abs(ak_bkg["By"]) < 1.6)
    & (ak_bkg["Bpt"] > 15.0)
    & (ak_bkg["Bpt"] < 50.0)
    & (ak_bkg["CentBin"] > 0)
    & (ak_bkg["CentBin"] < 90)
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
        return 0.0, 0.0, 0, 0, 0.0, 0.0

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
    best_weighted_signal = 0.0
    best_weighted_background = 0.0

    for cut in candidate_cuts:
        passed_signal = int(np.count_nonzero(signal_scores > cut))
        passed_background = int(np.count_nonzero(background_scores > cut))
        weighted_signal = SIGNAL_SCALE_FACTOR * passed_signal
        weighted_background = BACKGROUND_SCALE_FACTOR * passed_background
        total = weighted_signal + weighted_background
        if total <= 0:
            continue

        significance = weighted_signal / np.sqrt(total)
        if significance > best_significance:
            best_significance = significance
            best_cut = float(cut)
            best_signal = passed_signal
            best_background = passed_background
            best_weighted_signal = weighted_signal
            best_weighted_background = weighted_background

    return best_significance, best_cut, best_signal, best_background, best_weighted_signal, best_weighted_background


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
    (
        best_significance,
        best_cut,
        best_signal,
        best_background,
        best_weighted_signal,
        best_weighted_background,
    ) = best_significance_from_scores(sig, bkg)
    trial.set_user_attr("best_cut", best_cut)
    trial.set_user_attr("best_signal_yield", best_signal)
    trial.set_user_attr("best_background_yield", best_background)
    trial.set_user_attr("best_weighted_signal_yield", best_weighted_signal)
    trial.set_user_attr("best_weighted_background_yield", best_weighted_background)
    return best_significance


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=number_trials)

print("Best params:", study.best_params)
print(f"Best validation score cut for objective: {study.best_trial.user_attrs.get('best_cut', 0.0):.4f}")
print(f"Best validation raw S: {study.best_trial.user_attrs.get('best_signal_yield', 0)}")
print(f"Best validation raw B: {study.best_trial.user_attrs.get('best_background_yield', 0)}")
print(f"Best validation weighted S: {study.best_trial.user_attrs.get('best_weighted_signal_yield', 0.0):.6f}")
print(f"Best validation weighted B: {study.best_trial.user_attrs.get('best_weighted_background_yield', 0.0):.6f}")
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
print(f"Score plot saved to: {score_plot_path}")

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
print(f"Test ROC saved to: {roc_json_path}")

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
print(f"Test ROC plot saved to: {roc_plot_path}")

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
    optimization_metric="max validation weighted S/sqrt(S+B) from score-cut scan",
    best_objective_value=study.best_value,
    notes={
        "search_space_tag": search_space_tag,
        "best_validation_cut": float(study.best_trial.user_attrs.get("best_cut", 0.0)),
        "best_validation_score_cut_for_objective": float(study.best_trial.user_attrs.get("best_cut", 0.0)),
        "best_validation_signal_yield_raw": int(study.best_trial.user_attrs.get("best_signal_yield", 0)),
        "best_validation_background_yield_raw": int(study.best_trial.user_attrs.get("best_background_yield", 0)),
        "best_validation_signal_yield_weighted": float(study.best_trial.user_attrs.get("best_weighted_signal_yield", 0.0)),
        "best_validation_background_yield_weighted": float(study.best_trial.user_attrs.get("best_weighted_background_yield", 0.0)),
        "signal_scale_factor": float(SIGNAL_SCALE_FACTOR),
        "background_scale_factor": float(BACKGROUND_SCALE_FACTOR),
        "signal_window_width": float(SIGNAL_WINDOW_WIDTH),
        "sideband_window_width": float(SIDEBAND_WINDOW_WIDTH),
        "test_auc": float(roc_auc),
        "test_roc_json_path": roc_json_path,
        "test_roc_plot_path": roc_plot_path,
    },
)

print("Training complete. Model artifacts saved.")
