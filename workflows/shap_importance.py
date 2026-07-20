import argparse
import json


import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import uproot

from configs.samples import (
    bd_pbpb_precut_paths,
    infer_channel_from_tag,
    infer_dataset_year,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_training_config,
    supports_bd_pbpb_precut,
    to_root_spec,
)
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
from utils.selection import apply_selection

def parse_args():
    parser = argparse.ArgumentParser(description="Run SHAP analysis for a trained XGBoost model.")
    parser.add_argument("train_tag")
    parser.add_argument("max_events", nargs="?", type=int, default=20000)
    parser.add_argument("--use-precut", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


args = parse_args()
train_tag = args.train_tag
max_events = args.max_events
use_precut = bool(args.use_precut)
sample = infer_sample_from_tag(train_tag)
channel = infer_channel_from_tag(train_tag)
year = infer_dataset_year(train_tag, sample)
sel = infer_selection_profile(train_tag, sample)
train_cfg = resolve_training_config(sample, channel, year, sel)

SIG_PATH = to_root_spec(train_cfg["signal"])
BKG_PATH = to_root_spec(train_cfg["background"])
if use_precut:
    if not supports_bd_pbpb_precut(train_tag):
        raise ValueError(f"--use-precut only supports Bd_pb23/Bd_pb24 single-DAG tags, got '{train_tag}'.")
    train_background_path = bd_pbpb_precut_paths(train_tag)["train_background"]
    if not train_background_path.exists():
        raise FileNotFoundError(f"Missing precut training background file: {train_background_path}")
    BKG_PATH = str(train_background_path) + ":" + train_cfg["background"]["tree"]
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
df_sig = apply_selection(df_sig, train_cfg["signal_selection"], "signal_selection")
df_bkg = apply_selection(df_bkg, train_cfg["background_selection"], "background_selection")

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

fig, ax_left = plt.subplots(figsize=(8.8, 5.2))
x_pos = np.arange(len(ordered_names))
feature_percent = ordered_fractions * 100.0

# Left axis: cumulative contribution
line_cum, = ax_left.plot(x_pos, cumulative_percent, color="black", linewidth=1.2, alpha=0.8, label="Cumulative (%)")
ax_left.scatter(x_pos, cumulative_percent, color="tab:blue", s=45)
ax_left.axhline(95.0, color="tab:red", linestyle="--", linewidth=1.5, label="95%")
ax_left.set_ylim(0, 105)
ax_left.set_ylabel("Cumulative contribution (%)", color="tab:blue")
ax_left.tick_params(axis="y", labelcolor="tab:blue")

# Right axis: per-feature contribution from shap_bar (percentage form)
ax_right = ax_left.twinx()
bars = ax_right.bar(
    x_pos,
    feature_percent,
    width=0.68,
    color="tab:orange",
    alpha=0.35,
    edgecolor="tab:orange",
    label="Per-feature (%)",
)
right_max = float(np.max(feature_percent)) if len(feature_percent) else 0.0
# Right-axis limit rule: ceil(max/5%)*5% + one extra 5% bin
if right_max <= 0.0:
    right_ylim_top = 5.0
else:
    right_ylim_top = (np.ceil(right_max / 5.0) + 1.0) * 5.0
ax_right.set_ylim(0, right_ylim_top)
right_ticks = np.arange(0.0, right_ylim_top + 0.1, 5.0)
ax_right.set_yticks(right_ticks)
# Per-feature reference lines every 5%
for yv in right_ticks:
    if yv > 0.0:
        ax_right.axhline(yv, color="tab:orange", linestyle=":", linewidth=0.8, alpha=0.35, zorder=0)
ax_right.set_ylabel("Per-feature contribution (%)", color="tab:orange")
ax_right.tick_params(axis="y", labelcolor="tab:orange")

ax_left.set_xticks(x_pos)
ax_left.set_xticklabels(ordered_names, rotation=25, ha="right")

ax_left.legend([line_cum, bars], ["Cumulative (%)", "Per-feature (%)"], loc="upper left")
fig.tight_layout()
fig.savefig(shap_cumulative_path(train_tag))
plt.close(fig)

print("SHAP analysis complete.")
