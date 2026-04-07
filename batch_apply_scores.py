import json
import sys

import joblib
import pandas as pd
import uproot

from utils.paths import (
    data_output_path,
    ensure_dir,
    mc_output_path,
    resolve_model_config_path,
    resolve_model_path,
    resolve_scaler_path,
    selected_dir,
    train_group_tag,
)

if len(sys.argv) < 2:
    print("Usage: python3 batch_apply_scores.py <train_tag> [<train_tag> ...]")
    sys.exit(1)

train_tags = sys.argv[1:]
group_tag = train_group_tag(train_tags)

MC_INPUT = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC.root:ntmix"
DATA_INPUT = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root:ntmix"

extra_output_columns = ["BQvalue", "nSelectedChargedTracks", "CentBin", "Bpt", "By"]
keep_mc_isx3872 = True


def ordered_unique(columns):
    seen = set()
    output = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


def score_branch_name(train_tag):
    return f"xgb_score_{train_tag}"


print(f"Loading model artifacts for group: {group_tag}")
models = []
reference_input_columns = None
reference_trans_columns = None

for train_tag in train_tags:
    resolved_model_path = resolve_model_path(train_tag)
    resolved_scaler_path = resolve_scaler_path(train_tag)
    resolved_config_path = resolve_model_config_path(train_tag)

    xgbc = joblib.load(resolved_model_path)
    scaler = joblib.load(resolved_scaler_path)

    with open(resolved_config_path) as f:
        config = json.load(f)

    input_columns = config["input_columns"]
    trans_columns = config["trans_columns"]

    if reference_input_columns is None:
        reference_input_columns = input_columns
        reference_trans_columns = trans_columns
    else:
        if input_columns != reference_input_columns:
            raise ValueError(f"Input columns do not match within group: {train_tags}")
        if trans_columns != reference_trans_columns:
            raise ValueError(f"Transformed columns do not match within group: {train_tags}")

    print(f"  [{train_tag}] Model: {resolved_model_path}")
    print(f"  [{train_tag}] Scaler: {resolved_scaler_path}")

    models.append(
        {
            "train_tag": train_tag,
            "model": xgbc,
            "scaler": scaler,
            "input_columns": input_columns,
            "trans_columns": trans_columns,
            "score_column": score_branch_name(train_tag),
        }
    )

input_columns = reference_input_columns
trans_columns = reference_trans_columns

base_output_columns = ordered_unique(["Bmass"] + input_columns + extra_output_columns)
mc_branches = ordered_unique(base_output_columns + (["isX3872"] if keep_mc_isx3872 else []))
data_branches = base_output_columns


def score_dataframe(df, model_bundle, output_columns):
    df_trans = pd.DataFrame(
        model_bundle["scaler"].transform(df[model_bundle["input_columns"]]),
        columns=model_bundle["trans_columns"],
        index=df.index,
    )
    scores = model_bundle["model"].predict_proba(df_trans[model_bundle["trans_columns"]])[:, 1]
    print(
        f"  [{model_bundle['train_tag']}] Score range for {model_bundle['score_column']}: "
        f"[{scores.min():.4f}, {scores.max():.4f}]"
    )
    df_out = df[output_columns].copy()
    df_out[model_bundle["score_column"]] = scores
    return df_out


output_dir = ensure_dir(selected_dir(group_tag))
print(f"Writing grouped scored events to: {output_dir}")

print(f"\nProcessing MC once: {MC_INPUT}")
df_mc = uproot.concatenate(MC_INPUT, filter_name=mc_branches, library="pd")
print(f"  Loaded {len(df_mc)} events with branches: {mc_branches}")

mc_output_columns = ordered_unique(base_output_columns + (["isX3872"] if keep_mc_isx3872 else []))
df_mc_out = None
for model_bundle in models:
    df_scored = score_dataframe(df_mc, model_bundle, mc_output_columns)
    if df_mc_out is None:
        df_mc_out = df_scored
    else:
        df_mc_out[model_bundle["score_column"]] = df_scored[model_bundle["score_column"]]

mc_path = mc_output_path(group_tag)
with uproot.recreate(mc_path) as f:
    f["ntmix"] = {col: df_mc_out[col].values for col in df_mc_out.columns}
print(f"  Saved grouped MC to: {mc_path}")

print(f"\nProcessing DATA once: {DATA_INPUT}")
df_data = uproot.concatenate(DATA_INPUT, filter_name=data_branches, library="pd")
print(f"  Loaded {len(df_data)} events with branches: {data_branches}")

df_data_out = None
for model_bundle in models:
    df_scored = score_dataframe(df_data, model_bundle, base_output_columns)
    if df_data_out is None:
        df_data_out = df_scored
    else:
        df_data_out[model_bundle["score_column"]] = df_scored[model_bundle["score_column"]]

data_path = data_output_path(group_tag)
with uproot.recreate(data_path) as f:
    f["ntmix"] = {col: df_data_out[col].values for col in df_data_out.columns}
print(f"  Saved grouped DATA to: {data_path}")

print(f"\nAll grouped outputs saved in: {output_dir}")
