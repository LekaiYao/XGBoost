import json
import os
import sys

import joblib
import pandas as pd
import uproot

if len(sys.argv) != 2:
    print("Usage: python3 apply.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

MODEL_DIR = "./xgb_output"
MC_INPUT = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
DATA_INPUT = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"
OUTPUT_DIR = "./selected_events"

MC_CUT = "isX3872 == 1"
DATA_CUT = None

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


def score_dataframe(df):
    df_trans = pd.DataFrame(
        scaler.transform(df[input_columns]),
        columns=trans_columns,
        index=df.index,
    )
    scores = xgbc.predict_proba(df_trans[trans_columns])[:, 1]
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    df_out = df[["Bmass"] + input_columns].copy()
    df_out["xgb_score"] = scores
    return df_out


print(f"\nProcessing MC: {MC_INPUT}")
df_mc = uproot.concatenate(MC_INPUT, library="pd")
print(f"  Loaded {len(df_mc)} events")
if MC_CUT:
    df_mc = df_mc.query(MC_CUT)
    print(f"  After cut {MC_CUT}: {len(df_mc)} events")

df_mc_out = score_dataframe(df_mc)
os.makedirs(OUTPUT_DIR, exist_ok=True)
mc_output = os.path.join(OUTPUT_DIR, f"MC_with_score_{train_tag}.root")
with uproot.recreate(mc_output) as f:
    f["tree"] = {col: df_mc_out[col].values for col in df_mc_out.columns}
print(f"  Saved to: {mc_output}")

print(f"\nProcessing DATA: {DATA_INPUT}")
df_data = uproot.concatenate(DATA_INPUT, library="pd")
print(f"  Loaded {len(df_data)} events")
if DATA_CUT:
    df_data = df_data.query(DATA_CUT)
    print(f"  After cut {DATA_CUT}: {len(df_data)} events")

df_data_out = score_dataframe(df_data)
data_output = os.path.join(OUTPUT_DIR, f"DATA_with_score_{train_tag}.root")
with uproot.recreate(data_output) as f:
    f["tree"] = {col: df_data_out[col].values for col in df_data_out.columns}
print(f"  Saved to: {data_output}")

print(f"\nAll done! Output in: {OUTPUT_DIR}")
