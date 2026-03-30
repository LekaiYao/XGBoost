import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import uproot

if len(sys.argv) not in (2, 3):
    print("Usage: python3 shap_importance.py <train_tag> [max_events]")
    sys.exit(1)

train_tag = sys.argv[1]
max_events = int(sys.argv[2]) if len(sys.argv) == 3 else 20000

MODEL_DIR = "./xgb_output"
OUTPUT_DIR = "./xgb_output"
SIG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
BKG_PATH = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"
RNG_SEED = 42

print("Loading model artifacts...")
xgbc = joblib.load(os.path.join(MODEL_DIR, f"xgb_model_{train_tag}.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, f"scaler_{train_tag}.pkl"))

config_path = os.path.join(MODEL_DIR, f"model_config_{train_tag}.json")
with open(config_path) as f:
    config = json.load(f)

input_columns = config["input_columns"]
trans_columns = config["trans_columns"]

print(f"  Model loaded from: {MODEL_DIR}")
print(f"  Config loaded from: {config_path}")
print(f"  Input columns: {input_columns}")

print("Loading signal and background samples...")
df_sig = uproot.concatenate(SIG_PATH, library="pd")
df_bkg = uproot.concatenate(BKG_PATH, library="pd")

df_sig = df_sig[df_sig["isX3872"] == 1].copy()
df_bkg = df_bkg[
    ((df_bkg["Bmass"] < 3.83) & (df_bkg["Bmass"] > 3.75))
    | ((df_bkg["Bmass"] > 3.91) & (df_bkg["Bmass"] < 4.0))
].copy()

df_raw = pd.concat([df_sig, df_bkg], axis=0, ignore_index=True)
print(f"  Total events before sampling: {len(df_raw)}")

if max_events > 0 and len(df_raw) > max_events:
    df_raw = df_raw.sample(n=max_events, random_state=RNG_SEED).sort_index()
    print(f"  Sampled events for SHAP: {len(df_raw)}")
else:
    print(f"  Using all events for SHAP: {len(df_raw)}")

X_trans = pd.DataFrame(
    scaler.transform(df_raw[input_columns]),
    columns=trans_columns,
    index=df_raw.index,
)
X_display = df_raw[input_columns].copy()

print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgbc)
shap_values = explainer.shap_values(X_trans)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
total_mean_abs_shap = float(mean_abs_shap.sum())

importance_pairs = sorted(
    zip(input_columns, mean_abs_shap),
    key=lambda item: item[1],
    reverse=True,
)

ordered_names = [name for name, _ in importance_pairs]
ordered_scores = np.array([float(score) for _, score in importance_pairs])

if total_mean_abs_shap > 0.0:
    ordered_fractions = ordered_scores / total_mean_abs_shap
else:
    ordered_fractions = np.zeros_like(ordered_scores)

cumulative_percent = np.cumsum(ordered_fractions) * 100.0

print("SHAP importance ranking (mean |SHAP|):")
for rank, (name, score) in enumerate(importance_pairs, start=1):
    print(f"  {rank}. {name}: {score:.6f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
json_path = os.path.join(OUTPUT_DIR, f"shap_importance_{train_tag}.json")
with open(json_path, "w") as f:
    json.dump(
        [
            {"rank": rank, "feature": name, "mean_abs_shap": float(score)}
            for rank, (name, score) in enumerate(importance_pairs, start=1)
        ],
        f,
        indent=2,
    )
print(f"SHAP importance saved to: {json_path}")

fraction_path = os.path.join(OUTPUT_DIR, f"shap_importance_fraction_{train_tag}.json")
with open(fraction_path, "w") as f:
    json.dump(
        [
            {
                "rank": rank,
                "feature": name,
                "mean_abs_shap": float(score),
                "fraction": float(fraction),
                "percent": float(fraction * 100.0),
                "cumulative_percent": float(cumulative),
            }
            for rank, ((name, score), fraction, cumulative) in enumerate(
                zip(importance_pairs, ordered_fractions, cumulative_percent),
                start=1,
            )
        ],
        f,
        indent=2,
    )
print(f"SHAP importance fractions saved to: {fraction_path}")

summary_path = os.path.join(OUTPUT_DIR, f"shap_summary_{train_tag}.pdf")
plt.figure(figsize=(8, 5.5))
shap.summary_plot(
    shap_values,
    features=X_display,
    feature_names=input_columns,
    show=False,
)
plt.tight_layout()
plt.savefig(summary_path)
plt.close()
print(f"SHAP summary plot saved to: {summary_path}")

bar_path = os.path.join(OUTPUT_DIR, f"shap_bar_{train_tag}.pdf")
plt.figure(figsize=(8, 5.5))
shap.summary_plot(
    shap_values,
    features=X_display,
    feature_names=input_columns,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig(bar_path)
plt.close()
print(f"SHAP bar plot saved to: {bar_path}")

cumulative_path = os.path.join(OUTPUT_DIR, f"shap_cumulative_{train_tag}.pdf")
plt.figure(figsize=(8, 5))
plt.plot(ordered_names, cumulative_percent, color="black", linewidth=1, alpha=0.6)
plt.scatter(ordered_names, cumulative_percent, color="tab:blue", s=45)
plt.axhline(95.0, color="tab:red", linestyle="--", linewidth=1.5, label="95% cumulative SHAP")
plt.ylabel("Cumulative SHAP importance (%)")
plt.xlabel("Features ordered by mean |SHAP|")
plt.ylim(0, 105)
plt.xticks(rotation=25, ha="right")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(cumulative_path)
plt.close()
print(f"SHAP cumulative plot saved to: {cumulative_path}")

print("SHAP analysis complete.")
