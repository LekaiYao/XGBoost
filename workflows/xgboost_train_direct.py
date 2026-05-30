import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_token_from_tag,
    infer_dataset_year,
    infer_sample_from_tag as infer_sample_from_config,
    infer_selection_profile,
    resolve_training_config,
    to_root_spec,
)
from configs.direct_xgb_settings import DIRECT_XGB_PARAMS
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
from utils.varsets import get_varset_columns, infer_sample_from_tag, infer_varset_from_tag


def parse_args():
    parser = argparse.ArgumentParser(description="Direct XGBoost training without Optuna")
    parser.add_argument("train_tag")
    parser.add_argument("--dataset-year", choices=["2023", "2024"], default=None)
    parser.add_argument("--selection-profile", default=None)
    return parser.parse_args()


def save_feature_importance(model, feature_names, train_tag):
    importance_pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    with open(condor_feature_importance_path(train_tag), "w") as f:
        json.dump(
            [{"rank": i, "feature": n, "importance": float(s)} for i, (n, s) in enumerate(importance_pairs, 1)],
            f,
            indent=2,
        )

    names = [n for n, _ in importance_pairs]
    scores = np.array([float(s) for _, s in importance_pairs])
    cum = np.cumsum(scores) / np.sum(scores) * 100.0
    plt.figure(figsize=(8, 5))
    plt.plot(names, cum, color="black", linewidth=1, alpha=0.6)
    plt.scatter(names, cum, color="tab:blue", s=45)
    plt.axhline(95.0, color="tab:red", linestyle="--", linewidth=1.5)
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(condor_feature_importance_cumulative_path(train_tag))
    plt.close()


def main():
    args = parse_args()
    train_tag = args.train_tag

    sample = infer_sample_from_config(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset_token = infer_dataset_token_from_tag(train_tag)
    dataset_year = args.dataset_year or infer_dataset_year(train_tag, sample)
    selection_profile = args.selection_profile or infer_selection_profile(train_tag, sample)
    train_cfg = resolve_training_config(sample, channel, dataset_year, selection_profile)

    sample_key = infer_sample_from_tag(train_tag)
    feature_set_tag = infer_varset_from_tag(train_tag, sample=sample_key)
    if feature_set_tag is None:
        raise ValueError(f"Unable to infer feature set from train_tag: {train_tag}")
    input_columns = get_varset_columns(sample_key, feature_set_tag, channel=channel)
    trans_columns = [f"{col}_trans" for col in input_columns]

    sig_path = to_root_spec(train_cfg["signal"])
    bkg_path = to_root_spec(train_cfg["background"])
    signal_selection = train_cfg["signal_selection"]
    background_selection = train_cfg["background_selection"]

    print(f"Train tag: {train_tag}")
    print(f"Dataset source: {train_cfg['dataset_source']}")
    print(f"Selection profile: {selection_profile}")

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
    df_trans = pd.DataFrame(scaler.fit_transform(df_raw[input_columns]), columns=trans_columns, index=df_raw.index)
    df = pd.concat([df_trans, df_raw], axis=1)
    y = df[["is_sig", "is_bkg"]]

    X_train, X_test, y_train, y_test = train_test_split(df[trans_columns], y, train_size=0.75, random_state=42)

    n_sig_train = int(y_train["is_sig"].sum())
    n_bkg_train = int(y_train["is_bkg"].sum())
    if n_sig_train <= 0:
        raise ValueError("No signal events available in training split after selection.")
    scale_pos_weight = float(n_bkg_train / n_sig_train)

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
        "scale_pos_weight": scale_pos_weight,
        **DIRECT_XGB_PARAMS[dataset_token][channel],
    }
    xgbc = XGBClassifier(**params)
    xgbc.fit(X_train, y_train["is_sig"].to_numpy())

    test_score_xgb = xgbc.predict_proba(X_test)
    test_score_xgb_sig = test_score_xgb[y_test["is_sig"]][:, 1]
    test_score_xgb_bkg = test_score_xgb[y_test["is_bkg"]][:, 1]
    test_score_xgb_all = test_score_xgb[:, 1]
    test_y_true = y_test["is_sig"].astype(int).to_numpy()
    fpr, tpr, thresholds = roc_curve(test_y_true, test_score_xgb_all)
    roc_auc = auc(fpr, tpr)

    ensure_dir(condor_model_dir(train_tag))
    ensure_dir(condor_training_dir(train_tag))
    joblib.dump(xgbc, condor_model_path(train_tag))
    joblib.dump(scaler, condor_scaler_path(train_tag))

    with open(condor_model_config_path(train_tag), "w") as f:
        json.dump({"input_columns": input_columns, "trans_columns": trans_columns, "model_params": params}, f, indent=2)

    score_plot_path = condor_training_score_path(train_tag)
    plt.figure(figsize=(6, 6))
    plt.hist(test_score_xgb_sig, label="signal", histtype="step", bins=np.linspace(0, 1, 100), density=True)
    plt.hist(test_score_xgb_bkg, label="background", histtype="step", bins=np.linspace(0, 1, 100), density=True)
    plt.xlabel("Score (Prob. from XGBoost Prediction)")
    plt.ylabel("(Bin Width)$^{-1}$")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(score_plot_path)
    plt.close()

    roc_json_path = f"{condor_training_dir(train_tag)}/test_roc.json"
    with open(roc_json_path, "w") as f:
        json.dump(
            {
                "auc": float(roc_auc),
                "threshold": [float(x) for x in thresholds],
                "tpr": [float(x) for x in tpr],
                "fpr": [float(x) for x in fpr],
                "background_rejection": [float(1.0 - x) for x in fpr],
                "signal_efficiency": [float(x) for x in tpr],
                "mode": "direct_xgboost",
            },
            f,
            indent=2,
        )

    save_feature_importance(xgbc, input_columns, train_tag)

    save_run_metadata(
        train_tag=train_tag,
        training_script="workflows/xgboost_train_direct.py",
        signal_path=sig_path,
        background_path=bkg_path,
        signal_selection=train_cfg["signal_selection"],
        background_selection=train_cfg["background_selection"],
        input_columns=input_columns,
        trans_columns=trans_columns,
        pos_weight=scale_pos_weight,
        fixed_model_params=params,
        best_model_params=params,
        is_optuna=False,
        optimization_metric="direct training test AUC",
        best_objective_value=float(roc_auc),
        notes={
            "training_mode": "direct_xgboost",
            "dataset_source": train_cfg["dataset_source"],
            "dataset_year": dataset_year,
            "selection_profile": selection_profile,
            "sample": sample,
            "channel": channel,
            "auc": float(roc_auc),
            "n_sig_train": n_sig_train,
            "n_bkg_train": n_bkg_train,
            "scale_pos_weight": scale_pos_weight,
        },
    )

    print(f"Direct XGBoost training complete: {train_tag}, AUC={roc_auc:.4f}")


if __name__ == "__main__":
    main()
