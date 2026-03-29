import os
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import optuna

# =========================
# input
# =========================
if len(sys.argv) != 2:
    print("Usage: python3 optuna_XGBoost.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

number_trials=150 #3->50->150

SIG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
BKG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"

ak_sig = uproot.concatenate(SIG_PATH, library="pd")
ak_bkg = uproot.concatenate(BKG_PATH, library="pd")

# =========================
# selection
# =========================
ak_sig = ak_sig[(ak_sig["isX3872"] == 1)]

ak_bkg = ak_bkg[
    ((ak_bkg["Bmass"] < 3.83) & (ak_bkg["Bmass"] > 3.75)) |
    ((ak_bkg["Bmass"] > 3.91) & (ak_bkg["Bmass"] < 4.0))
]

ak_sig["is_sig"] = True
ak_bkg["is_sig"] = False
ak_sig["is_bkg"] = False
ak_bkg["is_bkg"] = True

df_raw = pd.concat([ak_sig, ak_bkg], axis=0, ignore_index=True)

# =========================
# feature scaling
# =========================
from sklearn.preprocessing import StandardScaler

input_columns = ['Btrk1dR', 'Btrk2dR', 'BtrkPtimb']
scaler = StandardScaler()

df_trans = pd.DataFrame(
    scaler.fit_transform(df_raw[input_columns]),
    columns=[c + '_trans' for c in input_columns],
    index=df_raw.index
)

df = pd.concat([df_trans, df_raw], axis=1)

# =========================
# dataset split
# =========================
X = df[['Btrk1dR_trans', 'Btrk2dR_trans', 'BtrkPtimb_trans']]
y = df[['is_sig', 'is_bkg']]

from sklearn.model_selection import train_test_split

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)

# =========================
# class weight
# =========================
n_sig = (y_train['is_sig'] == 1).sum()
n_bkg = (y_train['is_sig'] == 0).sum()
pos_weight = n_bkg / n_sig

# =========================
# XGBoost + Optuna
# =========================
from xgboost import XGBClassifier

def objective(trial):

    params = {
        "eval_metric": "logloss",
        "scale_pos_weight": pos_weight,

        # ===== tuned parameters =====
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train['is_sig'])

    pred = model.predict_proba(X_val)[:, 1]

    sig = pred[y_val['is_sig'] == 1]
    bkg = pred[y_val['is_sig'] == 0]

    score = np.mean(sig) - np.mean(bkg)

    return score

# =========================
# run optuna
# =========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=number_trials)

print("Best params:", study.best_params)

# =========================
# train final model
# =========================
xgbc = XGBClassifier(
    **study.best_params,
    eval_metric="logloss",
    scale_pos_weight=pos_weight
)

xgbc.fit(X_train, y_train['is_sig'])

# =========================
# output
# =========================
output_dir = "./xgb_output"
os.makedirs(output_dir, exist_ok=True)

test_score_xgb = xgbc.predict_proba(X_test)

test_score_xgb_sig = test_score_xgb[y_test['is_sig']][:, 1]
test_score_xgb_bkg = test_score_xgb[y_test['is_bkg']][:, 1]

plt.figure(figsize=(6, 6))
plt.hist(test_score_xgb_sig, label="X(3872)", histtype="step",
         bins=np.linspace(0, 1, 100), density=True)
plt.hist(test_score_xgb_bkg, label="bkg", histtype="step",
         bins=np.linspace(0, 1, 100), density=True)

plt.xlabel("Score (Prob. from XGBoost Prediction)")
plt.ylabel("(Bin Width)$^{-1}$")
plt.legend()
plt.xlim(0, 1)

plt.savefig(f"{output_dir}/xgb_score_{train_tag}.pdf")

# =========================
# save model
# =========================
import joblib
import json

joblib.dump(xgbc, f"{output_dir}/xgb_model_{train_tag}.pkl")
joblib.dump(scaler, f"{output_dir}/scaler_{train_tag}.pkl")

config = {
    "input_columns": input_columns,
    "trans_columns": [c + '_trans' for c in input_columns]
}

with open(f"{output_dir}/model_config_{train_tag}.json", "w") as f:
    json.dump(config, f, indent=2)

print("Training complete. Model artifacts saved.")