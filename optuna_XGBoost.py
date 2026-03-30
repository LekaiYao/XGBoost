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

if len(sys.argv) != 2:
    print("Usage: python3 optuna_XGBoost.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]
number_trials = int(os.environ.get("OPTUNA_N_TRIALS", "150"))

SIG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
BKG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"

input_columns = ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"]


def save_feature_importance(model, feature_names, output_dir, train_tag):
    importance_pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    print("Feature importance ranking:")
    for rank, (name, score) in enumerate(importance_pairs, start=1):
        print(f"  {rank}. {name}: {score:.6f}")

    importance_path = os.path.join(output_dir, f"feature_importance_{train_tag}.json")
    with open(importance_path, "w") as f:
        json.dump(
            [
                {"rank": rank, "feature": name, "importance": float(score)}
                for rank, (name, score) in enumerate(importance_pairs, start=1)
            ],
            f,
            indent=2,
        )
    print(f"Feature importance saved to: {importance_path}")

    ordered_names = [name for name, _ in importance_pairs]
    ordered_scores = np.array([float(score) for _, score in importance_pairs])
    cumulative_percent = np.cumsum(ordered_scores) / np.sum(ordered_scores) * 100.0

    plot_path = os.path.join(output_dir, f"feature_importance_cumulative_{train_tag}.pdf")
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
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {plot_path}")


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
        "eval_metric": "logloss",
        "scale_pos_weight": pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

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
    eval_metric="logloss",
    scale_pos_weight=pos_weight,
)
xgbc.fit(X_train, y_train["is_sig"])

output_dir = "./xgb_output"
os.makedirs(output_dir, exist_ok=True)

test_score_xgb = xgbc.predict_proba(X_test)
test_score_xgb_sig = test_score_xgb[y_test["is_sig"]][:, 1]
test_score_xgb_bkg = test_score_xgb[y_test["is_bkg"]][:, 1]

plt.figure(figsize=(6, 6))
plt.hist(test_score_xgb_sig, label="X(3872)", histtype="step", bins=np.linspace(0, 1, 100), density=True)
plt.hist(test_score_xgb_bkg, label="bkg", histtype="step", bins=np.linspace(0, 1, 100), density=True)
plt.xlabel("Score (Prob. from XGBoost Prediction)")
plt.ylabel("(Bin Width)$^{-1}$")
plt.legend()
plt.xlim(0, 1)
plt.savefig(f"{output_dir}/xgb_score_{train_tag}.pdf")
plt.close()

joblib.dump(xgbc, f"{output_dir}/xgb_model_{train_tag}.pkl")
joblib.dump(scaler, f"{output_dir}/scaler_{train_tag}.pkl")

config = {
    "input_columns": input_columns,
    "trans_columns": trans_columns,
}
with open(f"{output_dir}/model_config_{train_tag}.json", "w") as f:
    json.dump(config, f, indent=2)

save_feature_importance(xgbc, input_columns, output_dir, train_tag)

print("Training complete. Model artifacts saved.")
