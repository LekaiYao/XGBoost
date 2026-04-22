import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import auc, roc_curve
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
from utils.varsets import VARSET_COLUMNS, infer_varset_from_tag

if len(sys.argv) != 2:
    print("Usage: python3 XGBoost.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

SIG_PATH = "/afs/cern.ch/user/l/leyao/work/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
BKG_PATH = "/afs/cern.ch/user/l/leyao/work/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"

varset_key = infer_varset_from_tag(train_tag)
if not varset_key or varset_key not in VARSET_COLUMNS:
    supported = ", ".join(sorted(VARSET_COLUMNS.keys()))
    raise ValueError(
        f"Cannot infer varset from train_tag='{train_tag}'. "
        f"Expected one of: {supported}."
    )
input_columns = list(VARSET_COLUMNS[varset_key])
SIGNAL_SELECTION = "isX3872 == 1"
BACKGROUND_SELECTION = "(3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)"
FIXED_MODEL_PARAMS = {"eval_metric": "logloss"}
RANDOM_STATE = 42


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

ak_sig = uproot.concatenate(SIG_PATH, library="pd")
ak_bkg = uproot.concatenate(BKG_PATH, library="pd")

ak_sig = ak_sig[ak_sig["isX3872"] == 1]
ak_bkg = ak_bkg[
    ((ak_bkg["Bmass"] < 3.83) & (ak_bkg["Bmass"] > 3.75))
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y["is_sig"],
    random_state=RANDOM_STATE,
)

n_sig = (y_train["is_sig"] == 1).sum()
n_bkg = (y_train["is_sig"] == 0).sum()
pos_weight = n_bkg / n_sig

xgbc = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=pos_weight,
)

xgbc.fit(X_train, y_train["is_sig"])

train_score_xgb = xgbc.predict_proba(X_train)[:, 1]
test_score_xgb = xgbc.predict_proba(X_test)[:, 1]
train_sig_mask = y_train["is_sig"].to_numpy(dtype=bool)
test_sig_mask = y_test["is_sig"].to_numpy(dtype=bool)
train_score_xgb_sig = train_score_xgb[train_sig_mask]
train_score_xgb_bkg = train_score_xgb[~train_sig_mask]
test_score_xgb_sig = test_score_xgb[test_sig_mask]
test_score_xgb_bkg = test_score_xgb[~test_sig_mask]

score_plot_path = training_score_path(train_tag)
plt.figure(figsize=(6, 6))
bins = np.linspace(0, 1, 51)

def density_points_with_errors(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    total = counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * widths
    if total == 0:
        density = np.zeros_like(centers)
        stat_err = np.zeros_like(centers)
    else:
        density = counts / (total * widths)
        # Poisson statistical uncertainty for density in each bin.
        stat_err = np.sqrt(counts) / (total * widths)
    return centers, density, stat_err, xerr


blue = "tab:blue"
orange = "tab:orange"

plt.hist(
    test_score_xgb_sig,
    label=r"test X(3872)",
    bins=bins,
    density=True,
    color=blue,
    alpha=0.25,
    histtype="stepfilled",
    linewidth=0,
)
plt.hist(
    test_score_xgb_bkg,
    label=r"test bkg",
    bins=bins,
    density=True,
    color=orange,
    alpha=0.25,
    histtype="stepfilled",
    linewidth=0,
)

train_sig_x, train_sig_y, train_sig_err, train_sig_xerr = density_points_with_errors(train_score_xgb_sig, bins)
train_bkg_x, train_bkg_y, train_bkg_err, train_bkg_xerr = density_points_with_errors(train_score_xgb_bkg, bins)
plt.errorbar(
    train_sig_x,
    train_sig_y,
    xerr=train_sig_xerr,
    yerr=train_sig_err,
    fmt="o",
    ms=0.15,
    color=blue,
    elinewidth=1,
    capsize=2,
    label=r"train X(3872)",
)
plt.errorbar(
    train_bkg_x,
    train_bkg_y,
    xerr=train_bkg_xerr,
    yerr=train_bkg_err,
    fmt="o",
    ms=0.15,
    color=orange,
    elinewidth=1,
    capsize=2,
    label=r"train bkg",
)
plt.xlabel("Score (Prob. from XGBoost Prediction)")
plt.ylabel("(Bin Width)$^{-1}$")
plt.legend()
plt.xlim(0, 1)
plt.savefig(score_plot_path)
plt.close()
print(f"Score plot saved to: {score_plot_path}")

roc_plot_path = os.path.join(training_dir(train_tag), "ROC.pdf")
roc_json_path = os.path.join(training_dir(train_tag), "test_roc.json")
train_fpr, train_tpr, train_thresholds = roc_curve(y_train["is_sig"].astype(int).to_numpy(), train_score_xgb)
test_fpr, test_tpr, test_thresholds = roc_curve(y_test["is_sig"].astype(int).to_numpy(), test_score_xgb)
train_auc = auc(train_fpr, train_tpr)
test_auc = auc(test_fpr, test_tpr)

with open(roc_json_path, "w") as f:
    json.dump(
        {
            "train_auc": float(train_auc),
            "test_auc": float(test_auc),
            "train_curve": {
                "fpr": train_fpr.tolist(),
                "tpr": train_tpr.tolist(),
                "thresholds": train_thresholds.tolist(),
            },
            "test_curve": {
                "fpr": test_fpr.tolist(),
                "tpr": test_tpr.tolist(),
                "thresholds": test_thresholds.tolist(),
            },
        },
        f,
        indent=2,
    )
print(f"ROC json saved to: {roc_json_path}")

plt.figure(figsize=(6, 6))
train_rej = 1.0 - train_fpr
test_rej = 1.0 - test_fpr
plt.plot(train_tpr, train_rej, linewidth=2, color="tab:blue", label="train")
plt.plot(test_tpr, test_rej, linewidth=2, linestyle="--", color="tab:orange", label="test")
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title(f"ROC: {train_tag}")
plt.text(
    0.03,
    0.05,
    f"train AUC = {train_auc:.4f}\ntest AUC = {test_auc:.4f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    ha="left",
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(roc_plot_path)
plt.close()
print(f"ROC plot saved to: {roc_plot_path}")

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
    training_script="XGBoost.py",
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
        "random_state": RANDOM_STATE,
    },
    best_model_params={
        **FIXED_MODEL_PARAMS,
        "scale_pos_weight": float(pos_weight),
        "random_state": RANDOM_STATE,
    },
)

print("Training complete. Model artifacts saved.")
