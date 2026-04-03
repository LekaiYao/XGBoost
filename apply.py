import json
import sys

import joblib
import pandas as pd
import uproot

from paths import (
    data_output_path,
    ensure_dir,
    mc_output_path,
    resolve_model_config_path,
    resolve_model_path,
    resolve_scaler_path,
    selected_dir,
)

if len(sys.argv) != 2:
    print("Usage: python3 apply.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

MC_INPUT = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
DATA_INPUT = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"

#MC_CUT = "isX3872 == 1"
MC_CUT = None
DATA_CUT = None

print("Loading model artifacts...")
resolved_model_path = resolve_model_path(train_tag)
resolved_scaler_path = resolve_scaler_path(train_tag)
resolved_config_path = resolve_model_config_path(train_tag)

xgbc = joblib.load(resolved_model_path)
scaler = joblib.load(resolved_scaler_path)

with open(resolved_config_path) as f:
    config = json.load(f)

input_columns = config["input_columns"]
trans_columns = config["trans_columns"]

print(f"  Model loaded from: {resolved_model_path}")
print(f"  Scaler loaded from: {resolved_scaler_path}")
print(f"  Config loaded from: {resolved_config_path}")
print(f"  Input columns: {input_columns}")

# Add any extra branches you want to keep in the output ROOT here.
# Example:
# extra_output_columns = ["Bpt", "Beta", "BsvpvDistance"]
# They will be written together with Bmass, the training inputs, and xgb_score.
extra_output_columns = ["BQvalue","nSelectedChargedTracks","CentBin","Bpt","By","BLxy"]

# Toggle whether MC output keeps the truth label branch.
keep_mc_isx3872 = True

def score_dataframe(df, extra_columns=None):
    df_trans = pd.DataFrame(
        scaler.transform(df[input_columns]),
        columns=trans_columns,
        index=df.index,
    )
    scores = xgbc.predict_proba(df_trans[trans_columns])[:, 1]
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    if extra_columns is None:
        extra_columns = []

    output_columns = ["Bmass"] + input_columns + extra_output_columns + extra_columns
    df_out = df[output_columns].copy()
    df_out["xgb_score"] = scores
    return df_out


output_dir = ensure_dir(selected_dir(train_tag))
print(f"Writing scored events to: {output_dir}")

print(f"\nProcessing MC: {MC_INPUT}")
df_mc = uproot.concatenate(MC_INPUT, library="pd")
print(f"  Loaded {len(df_mc)} events")
if MC_CUT:
    df_mc = df_mc.query(MC_CUT)
    print(f"  After cut {MC_CUT}: {len(df_mc)} events")

mc_extra_columns = ["isX3872"] if keep_mc_isx3872 else []
df_mc_out = score_dataframe(df_mc, extra_columns=mc_extra_columns)
mc_path = mc_output_path(train_tag)
with uproot.recreate(mc_path) as f:
    f["tree"] = {col: df_mc_out[col].values for col in df_mc_out.columns}
print(f"  Saved to: {mc_path}")

print(f"\nProcessing DATA: {DATA_INPUT}")
df_data = uproot.concatenate(DATA_INPUT, library="pd")
print(f"  Loaded {len(df_data)} events")
if DATA_CUT:
    df_data = df_data.query(DATA_CUT)
    print(f"  After cut {DATA_CUT}: {len(df_data)} events")

df_data_out = score_dataframe(df_data)
data_path = data_output_path(train_tag)
with uproot.recreate(data_path) as f:
    f["tree"] = {col: df_data_out[col].values for col in df_data_out.columns}
print(f"  Saved to: {data_path}")

print(f"\nAll done! Output in: {output_dir}")
