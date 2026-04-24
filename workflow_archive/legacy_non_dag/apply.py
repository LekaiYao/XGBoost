import json
import sys

import joblib
import pandas as pd
import uproot

from utils.paths import (
    ensure_dir,
    resolve_model_config_path,
    resolve_model_path,
    resolve_scaler_path,
    selected_dir,
)

if len(sys.argv) != 2:
    print("Usage: python3 apply.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

DATA_INPUT_FILE = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root"
DATA_INPUT_TREE = "ntmix"
DATA_OUTPUT_FILE = "DATA_wScore.root"

MC_INPUTS = [
    {
        "input_file": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_PSI2S_nonPrompt.root",
        "input_tree": "ntmix_PSI2S",
        "output_file": "MC_PSI2S_nonPrompt_wScore.root",
    },
    {
        "input_file": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_PSI2S.root",
        "input_tree": "ntmix_PSI2S",
        "output_file": "MC_PSI2S_wScore.root",
    },
    {
        "input_file": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872_nonPrompt.root",
        "input_tree": "ntmix_X3872",
        "output_file": "MC_X3872_nonPrompt_wScore.root",
    },
    {
        "input_file": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872.root",
        "input_tree": "ntmix_X3872",
        "output_file": "MC_X3872_wScore.root",
    },
]

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


def ordered_unique(columns):
    seen = set()
    output = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


def score_dataframe(df):
    df_trans = pd.DataFrame(
        scaler.transform(df[input_columns]),
        columns=trans_columns,
        index=df.index,
    )
    scores = xgbc.predict_proba(df_trans[trans_columns])[:, 1]
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    df_out = df.copy()
    df_out["xgb_score"] = scores
    return df_out


output_dir = ensure_dir(selected_dir(train_tag))
print(f"Writing scored events to: {output_dir}")

for mc_spec in MC_INPUTS:
    mc_input_file = mc_spec["input_file"]
    mc_input_tree = mc_spec["input_tree"]
    mc_output_file = mc_spec["output_file"]
    mc_output_path = f"{output_dir}/{mc_output_file}"

    print(f"\nProcessing MC: {mc_input_file}:{mc_input_tree}")
    df_mc = uproot.open(mc_input_file)[mc_input_tree].arrays(library="pd")
    print(f"  Loaded {len(df_mc)} events")
    if MC_CUT:
        df_mc = df_mc.query(MC_CUT)
        print(f"  After cut {MC_CUT}: {len(df_mc)} events")

    df_mc_out = score_dataframe(df_mc)
    with uproot.recreate(mc_output_path) as f:
        f[mc_input_tree] = {col: df_mc_out[col].values for col in df_mc_out.columns}
    print(f"  Saved to: {mc_output_path}")

print(f"\nProcessing DATA: {DATA_INPUT_FILE}:{DATA_INPUT_TREE}")
df_data = uproot.open(DATA_INPUT_FILE)[DATA_INPUT_TREE].arrays(library="pd")
print(f"  Loaded {len(df_data)} events")
if DATA_CUT:
    df_data = df_data.query(DATA_CUT)
    print(f"  After cut {DATA_CUT}: {len(df_data)} events")

df_data_out = score_dataframe(df_data)
data_output_path = f"{output_dir}/{DATA_OUTPUT_FILE}"
with uproot.recreate(data_output_path) as f:
    f[DATA_INPUT_TREE] = {col: df_data_out[col].values for col in df_data_out.columns}
print(f"  Saved to: {data_output_path}")

print(f"\nAll done! Output in: {output_dir}")
