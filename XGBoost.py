import os
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Get training tag from command line
if len(sys.argv) != 2:
    print("Usage: python3 XGBoost.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

# Load with Awkward: jets and leptons are per-event variable-length vectors (jagged arrays).
# SIG_PATH = "/eos/user/c/coli/cms-repo/cmsdas/data/train_and_test_sig_hhbbww_1M.root:tree"
# BKG_PATH = "/eos/user/c/coli/cms-repo/cmsdas/data/train_and_test_bkg_ttbar_1M.root:tree"
SIG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
BKG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"

ak_sig = uproot.concatenate(SIG_PATH, library="pd")
ak_bkg = uproot.concatenate(BKG_PATH, library="pd")

# Signal: isX3872 == 1
mask_sig = (
    (ak_sig["isX3872"] == 1)
)
ak_sig = ak_sig[mask_sig]

# Background: sideband selection
mask_bkg = (
    ((ak_bkg["Bmass"] < 3.83) & (ak_bkg["Bmass"] > 3.75)) |
    ((ak_bkg["Bmass"] > 3.91) & (ak_bkg["Bmass"] < 4.0))
)
ak_bkg = ak_bkg[mask_bkg]

# add a column to indicate the signal or background
ak_sig["is_sig"] = True
ak_bkg["is_sig"] = False
ak_sig["is_bkg"] = False
ak_bkg["is_bkg"] = True

# concatenate the two dataframes
df_raw = pd.concat([ak_sig, ak_bkg], axis=0, ignore_index=True)

# Initialize the StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

input_columns = ['Btrk1dR', 'Btrk2dR', 'BtrkPtimb', 'Bchi2Prob']

# Fit and transform the DataFrame
df_trans = pd.DataFrame(scaler.fit_transform(df_raw[input_columns]), columns=[c + '_trans' for c in input_columns], index=df_raw.index)
df = pd.concat([df_trans, df_raw], axis=1)

# Define the features and labels
X = df[['Btrk1dR_trans', 'Btrk2dR_trans', 'BtrkPtimb_trans', 'Bchi2Prob_trans']]
y = df[['is_sig', 'is_bkg']] 

from sklearn.model_selection import train_test_split

# First, split the original dataset into a training set (80%) and a temporary validation/test set (20%)
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2)

# The final split ratio is: training/validation/test = 80/10/10
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)


# Import the XGBoost classifier
from xgboost import XGBClassifier

# Compute class weight
n_sig = (y_train['is_sig'] == 1).sum()
n_bkg = (y_train['is_sig'] == 0).sum()

pos_weight = n_bkg / n_sig

# Initialize the XGBoost classifier with logloss as the evaluation metric
# and fit it to the training data (X_train, y_train['is_sig'])
xgbc = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=pos_weight
)

xgbc.fit(X_train, y_train['is_sig'])

# output
output_dir = "./xgb_output"
os.makedirs(output_dir, exist_ok=True)

# Get the predicted probabilities for the test set
test_score_xgb = xgbc.predict_proba(X_test)

# Select the predicted probability for the signal class (class 1) for signal and background events
test_score_xgb_sig = test_score_xgb[y_test['is_sig']][:, 1]
test_score_xgb_bkg = test_score_xgb[y_test['is_bkg']][:, 1]

# Plot the distribution of XGBoost scores for signal and background events
plt.figure(figsize=(6, 6))
hist_sig = plt.hist(test_score_xgb_sig, label=r"X(3872)", histtype="step", bins=np.linspace(0, 1, 100), density=True)
hist_bkg = plt.hist(test_score_xgb_bkg, label=r"bkg", histtype="step", bins=np.linspace(0, 1, 100), density=True)
plt.xlabel("Score (Prob. from XGBoost Prediction)")
plt.ylabel("(Bin Width)$^{-1}$")
plt.legend()
plt.xlim(0, 1)
plt.savefig(f"{output_dir}/xgb_score_{train_tag}.pdf")

# save train result
import joblib

model_path = os.path.join(output_dir, f"xgb_model_{train_tag}.pkl")
joblib.dump(xgbc, model_path)
print(f"Model saved to: {model_path}")

scaler_path = os.path.join(output_dir, f"scaler_{train_tag}.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

import json
config = {
    "input_columns": input_columns,
    "trans_columns": [c + '_trans' for c in input_columns]
}
config_path = os.path.join(output_dir, f"model_config_{train_tag}.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_path}")

print("Training complete. Model artifacts saved.")