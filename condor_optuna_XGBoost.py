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
BKG_PATH = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root:ntmix"

input_columns = ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"]
SIGNAL_SELECTION = "isX3872 == 1"
BACKGROUND_SELECTION = "(3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)"
FIXED_MODEL_PARAMS = {"eval_metric": "logloss"}

SEARCH_SPACE_PRESETS = {
    "v1": {
        "n_estimators": {"type": "int", "low": 400, "high": 1400},
        "learning_rate": {"type": "float", "low": 0.03, "high": 0.12, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "min_child_weight": {"type": "int", "low": 2, "high": 8},
        "subsample": {"type": "float", "low": 0.75, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 2.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 6.0},
    },
    "v2": {
        "n_estimators": {"type": "int", "low": 600, "high": 1800},
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.08, "log": True},
        "max_depth": {"type": "int", "low": 4, "high": 6},
        "min_child_weight": {"type": "int", "low": 1, "high": 6},
        "subsample": {"type": "float", "low": 0.8, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.8, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 5.0},
    },
    "v3": {
        "n_estimators": {"type": "int", "low": 250, "high": 900},
        "learning_rate": {"type": "float", "low": 0.10, "high": 0.22},
        "max_depth": {"type": "int", "low": 4, "high": 7},
        "min_child_weight": {"type": "int", "low": 1, "high": 4},
        "subsample": {"type": "float", "low": 0.7, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.7, "high": 0.95},
        "gamma": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 4.0},
    },
    "v4": {
        "n_estimators": {"type": "int", "low": 300, "high": 1200},
        "learning_rate": {"type": "float", "low": 0.05, "high": 0.18},
        "max_depth": {"type": "int", "low": 5, "high": 7},
        "min_child_weight": {"type": "int", "low": 3, "high": 10},
        "subsample": {"type": "float", "low": 0.8, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.8, "high": 1.0},
        "gamma": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 2.0},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 8.0},
    },
    "v5": {
        "n_estimators": {"type": "int", "low": 500, "high": 1600},
        "learning_rate": {"type": "float", "low": 0.025, "high": 0.09, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 6},
        "min_child_weight": {"type": "int", "low": 4, "high": 12},
        "subsample": {"type": "float", "low": 0.6, "high": 0.85},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 0.85},
        "gamma": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_lambda": {"type": "float", "low": 2.0, "high": 10.0},
    },
    "v6": {
        "n_estimators": {"type": "int", "low": 250, "high": 1000},
        "learning_rate": {"type": "float", "low": 0.08, "high": 0.20},
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "min_child_weight": {"type": "int", "low": 1, "high": 5},
        "subsample": {"type": "float", "low": 0.9, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.9, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 0.8},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.5},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 3.0},
    },
    "v7": {
        "n_estimators": {"type": "int", "low": 700, "high": 2200},
        "learning_rate": {"type": "float", "low": 0.015, "high": 0.06, "log": True},
        "max_depth": {"type": "int", "low": 4, "high": 7},
        "min_child_weight": {"type": "int", "low": 1, "high": 4},
        "subsample": {"type": "float", "low": 0.7, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.7, "high": 0.95},
        "gamma": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 5.0},
    },
    "v8": {
        "n_estimators": {"type": "int", "low": 300, "high": 1100},
        "learning_rate": {"type": "float", "low": 0.12, "high": 0.25},
        "max_depth": {"type": "int", "low": 5, "high": 8},
        "min_child_weight": {"type": "int", "low": 1, "high": 3},
        "subsample": {"type": "float", "low": 0.75, "high": 0.95},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 0.95},
        "gamma": {"type": "float", "low": 0.0, "high": 0.6},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 0.4},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 2.5},
    },
    "v9": {
        "n_estimators": {"type": "int", "low": 350, "high": 1300},
        "learning_rate": {"type": "float", "low": 0.04, "high": 0.14},
        "max_depth": {"type": "int", "low": 5, "high": 6},
        "min_child_weight": {"type": "int", "low": 6, "high": 16},
        "subsample": {"type": "float", "low": 0.75, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.75, "high": 1.0},
        "gamma": {"type": "float", "low": 1.0, "high": 4.0},
        "reg_alpha": {"type": "float", "low": 0.5, "high": 3.0},
        "reg_lambda": {"type": "float", "low": 3.0, "high": 12.0},
    },
    "v10": {
        "n_estimators": {"type": "int", "low": 250, "high": 700},
        "learning_rate": {"type": "float", "low": 0.14, "high": 0.22},
        "max_depth": {"type": "int", "low": 5, "high": 6},
        "min_child_weight": {"type": "int", "low": 1, "high": 5},
        "subsample": {"type": "float", "low": 0.75, "high": 0.9},
        "colsample_bytree": {"type": "float", "low": 0.8, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 1.5},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1.0, "high": 5.0},
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
    ((ak_bkg["Bmass"] < 3.83) & (ak_bkg["Bmass"] > 3.75))
    | ((ak_bkg["Bmass"] > 3.91) & (ak_bkg["Bmass"] < 4.0))
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

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)

n_sig = (y_train["is_sig"] == 1).sum()
n_bkg = (y_train["is_sig"] == 0).sum()
pos_weight = n_bkg / n_sig


def objective(trial):
    params = {
        "eval_metric": FIXED_MODEL_PARAMS["eval_metric"],
        "scale_pos_weight": pos_weight,
    }
    for name, config in OPTUNA_SEARCH_SPACE.items():
        params[name] = suggest_param(trial, name, config)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train["is_sig"])

    pred = model.predict_proba(X_val)[:, 1]
    sig = pred[y_val["is_sig"] == 1]
    bkg = pred[y_val["is_sig"] == 0]
    return np.mean(sig) - np.mean(bkg)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=number_trials)

print("Best params:", study.best_params)

xgbc = XGBClassifier(
    **study.best_params,
    eval_metric=FIXED_MODEL_PARAMS["eval_metric"],
    scale_pos_weight=pos_weight,
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
    },
    best_model_params={
        **{key: (float(value) if isinstance(value, float) else value) for key, value in study.best_params.items()},
        **FIXED_MODEL_PARAMS,
        "scale_pos_weight": float(pos_weight),
    },
    is_optuna=True,
    optuna_n_trials=number_trials,
    optimized_hyperparameters=list(OPTUNA_SEARCH_SPACE.keys()),
    hyperparameter_search_space=OPTUNA_SEARCH_SPACE,
    optimization_metric="mean(sig_score) - mean(bkg_score)",
    best_objective_value=study.best_value,
    notes={"search_space_tag": search_space_tag},
)

print("Training complete. Model artifacts saved.")
