import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import uproot

from configs.samples import infer_dataset_year, infer_sample_from_tag, infer_selection_profile, resolve_training_config, to_root_spec
from utils.paths import (
    ensure_dir,
    resolve_model_config_path,
    resolve_model_path,
    resolve_scaler_path,
    shap_bar_path,
    shap_cumulative_path,
    shap_dir,
    shap_importance_fraction_path,
    shap_importance_path,
    shap_summary_path,
)

if len(sys.argv) not in (2, 3):
    print("Usage: python3 workflows/shap_importance.py <train_tag> [max_events]")
    sys.exit(1)

train_tag = sys.argv[1]
max_events = int(sys.argv[2]) if len(sys.argv) == 3 else 20000
sample = infer_sample_from_tag(train_tag)
year = infer_dataset_year(train_tag, sample)
sel = infer_selection_profile(train_tag, sample)
train_cfg = resolve_training_config(sample, year, sel)

SIG_PATH = to_root_spec(train_cfg["signal"])
BKG_PATH = to_root_spec(train_cfg["background"])
RNG_SEED = 42

resolved_model_path = resolve_model_path(train_tag)
resolved_scaler_path = resolve_scaler_path(train_tag)
resolved_config_path = resolve_model_config_path(train_tag)

xgbc = joblib.load(resolved_model_path)
scaler = joblib.load(resolved_scaler_path)
with open(resolved_config_path) as f:
    config = json.load(f)

input_columns = config["input_columns"]
trans_columns = config["trans_columns"]

df_sig = uproot.concatenate(SIG_PATH, library="pd")
df_bkg = uproot.concatenate(BKG_PATH, library="pd")
if "Bmass" in df_bkg.columns:
    df_bkg = df_bkg[
        ((df_bkg["Bmass"] < 3.83) & (df_bkg["Bmass"] > 3.75))
        | ((df_bkg["Bmass"] > 3.91) & (df_bkg["Bmass"] < 4.0))
    ].copy()

df_raw = pd.concat([df_sig, df_bkg], axis=0, ignore_index=True)
if max_events > 0 and len(df_raw) > max_events:
    df_raw = df_raw.sample(n=max_events, random_state=RNG_SEED).sort_index()

X_trans = pd.DataFrame(scaler.transform(df_raw[input_columns]), columns=trans_columns, index=df_raw.index)
X_display = df_raw[input_columns].copy()

explainer = shap.TreeExplainer(xgbc)
shap_values = explainer.shap_values(X_trans)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
total_mean_abs_shap = float(mean_abs_shap.sum())
importance_pairs = sorted(zip(input_columns, mean_abs_shap), key=lambda x: x[1], reverse=True)
ordered_names = [name for name, _ in importance_pairs]
ordered_scores = np.array([float(score) for _, score in importance_pairs])
ordered_fractions = ordered_scores / total_mean_abs_shap if total_mean_abs_shap > 0.0 else np.zeros_like(ordered_scores)
cumulative_percent = np.cumsum(ordered_fractions) * 100.0

ensure_dir(shap_dir(train_tag))
with open(shap_importance_path(train_tag), "w") as f:
    json.dump([{"rank": i, "feature": n, "mean_abs_shap": float(s)} for i, (n, s) in enumerate(importance_pairs, 1)], f, indent=2)
with open(shap_importance_fraction_path(train_tag), "w") as f:
    json.dump([
        {
            "rank": i,
            "feature": n,
            "mean_abs_shap": float(s),
            "fraction": float(fr),
            "percent": float(fr * 100.0),
            "cumulative_percent": float(cp),
        }
        for i, ((n, s), fr, cp) in enumerate(zip(importance_pairs, ordered_fractions, cumulative_percent), 1)
    ], f, indent=2)

plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, features=X_display, feature_names=input_columns, show=False)
plt.tight_layout(); plt.savefig(shap_summary_path(train_tag)); plt.close()

plt.figure(figsize=(8, 5.5))
shap.summary_plot(shap_values, features=X_display, feature_names=input_columns, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig(shap_bar_path(train_tag)); plt.close()

plt.figure(figsize=(8, 5))
plt.plot(ordered_names, cumulative_percent, color="black", linewidth=1, alpha=0.6)
plt.scatter(ordered_names, cumulative_percent, color="tab:blue", s=45)
plt.axhline(95.0, color="tab:red", linestyle="--", linewidth=1.5)
plt.ylim(0, 105)
plt.xticks(rotation=25, ha="right")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(shap_cumulative_path(train_tag)); plt.close()

print("SHAP analysis complete.")
