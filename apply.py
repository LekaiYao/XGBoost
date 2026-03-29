import os
import uproot
import pandas as pd
import numpy as np
import joblib
import sys

# Get training tag
if len(sys.argv) != 2:
    print("Usage: python3 apply.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

# path
MODEL_DIR = "./xgb_output"
MC_INPUT = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix"
DATA_INPUT = "/eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix"
OUTPUT_DIR = "./selected_events"

# cut
MC_CUT = "isX3872 == 1"      # e.g. "isX3872 == 1" 
DATA_CUT = None    # e.g. "(Bmass > 3.75 & Bmass < 3.83) | (Bmass > 3.91 & Bmass < 4.0)"

# ============================================
# load model
# ============================================

print("Loading model artifacts...")
xgbc = joblib.load(os.path.join(MODEL_DIR, f"xgb_model_{train_tag}.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, f"scaler_{train_tag}.pkl"))

# input
input_columns = ['Btrk1dR', 'Btrk2dR', 'BtrkPtimb']

print(f"  Model loaded from: {MODEL_DIR}")
print(f"  Input columns: {input_columns}")

# ============================================
# apply to MC
# ============================================

print(f"\nProcessing MC: {MC_INPUT}")
df_mc = uproot.concatenate(MC_INPUT, library="pd")
print(f"  Loaded {len(df_mc)} events")

if MC_CUT:
    df_mc = df_mc.query(MC_CUT)
    print(f"  After cut '{MC_CUT}': {len(df_mc)} events")

df_mc_trans = pd.DataFrame(
    scaler.transform(df_mc[input_columns]),
    columns=[c + '_trans' for c in input_columns],
    index=df_mc.index
)

# score
X_mc = df_mc_trans[[c + '_trans' for c in input_columns]]
mc_scores = xgbc.predict_proba(X_mc)[:, 1]
print(f"  Score range: [{mc_scores.min():.4f}, {mc_scores.max():.4f}]")

# output
df_mc_out = df_mc[['Bmass'] + input_columns].copy()
df_mc_out['xgb_score'] = mc_scores

os.makedirs(OUTPUT_DIR, exist_ok=True)
mc_output = os.path.join(OUTPUT_DIR, f"MC_with_score_{train_tag}.root")
with uproot.recreate(mc_output) as f:
    f["tree"] = {col: df_mc_out[col].values for col in df_mc_out.columns}
print(f"  Saved to: {mc_output}")

# ============================================
# apply to data
# ============================================

print(f"\nProcessing DATA: {DATA_INPUT}")
df_data = uproot.concatenate(DATA_INPUT, library="pd")
print(f"  Loaded {len(df_data)} events")

if DATA_CUT:
    df_data = df_data.query(DATA_CUT)
    print(f"  After cut '{DATA_CUT}': {len(df_data)} events")

df_data_trans = pd.DataFrame(
    scaler.transform(df_data[input_columns]),
    columns=[c + '_trans' for c in input_columns],
    index=df_data.index
)

# score
X_data = df_data_trans[[c + '_trans' for c in input_columns]]
data_scores = xgbc.predict_proba(X_data)[:, 1]
print(f"  Score range: [{data_scores.min():.4f}, {data_scores.max():.4f}]")

# output
df_data_out = df_data[['Bmass'] + input_columns].copy()
df_data_out['xgb_score'] = data_scores

data_output = os.path.join(OUTPUT_DIR, f"DATA_with_score_{train_tag}.root")
with uproot.recreate(data_output) as f:
    f["tree"] = {col: df_data_out[col].values for col in df_data_out.columns}
print(f"  Saved to: {data_output}")

print(f"\nAll done! Output in: {OUTPUT_DIR}")