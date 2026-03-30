import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if len(sys.argv) != 2:
    print("Usage: python3 XGBoost.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

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

xgbc = XGBClassifier(
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
plt.hist(test_score_xgb_sig, label=r"X(3872)", histtype="step", bins=np.linspace(0, 1, 100), density=True)
plt.hist(test_score_xgb_bkg, label=r"bkg", histtype="step", bins=np.linspace(0, 1, 100), density=True)
plt.xlabel("Score (Prob. from XGBoost Prediction)")
plt.ylabel("(Bin Width)$^{-1}$")
plt.legend()
plt.xlim(0, 1)
plt.savefig(f"{output_dir}/xgb_score_{train_tag}.pdf")

model_path = os.path.join(output_dir, f"xgb_model_{train_tag}.pkl")
joblib.dump(xgbc, model_path)
print(f"Model saved to: {model_path}")

scaler_path = os.path.join(output_dir, f"scaler_{train_tag}.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

config = {
    "input_columns": input_columns,
    "trans_columns": trans_columns,
}
config_path = os.path.join(output_dir, f"model_config_{train_tag}.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_path}")

save_feature_importance(xgbc, input_columns, output_dir, train_tag)

print("Training complete. Model artifacts saved.")
