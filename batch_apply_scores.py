import json
import os
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
    train_group_tag,
)

if len(sys.argv) < 2:
    print(
        "Usage: python3 batch_apply_scores.py "
        "[--output-tag <output_tag>] [--data-input <root:tree>] [--output-prefix <prefix>] [--dataset-year <2023|2024>] "
        "<train_tag> [<train_tag> ...]"
    )
    sys.exit(1)

args = sys.argv[1:]
output_tag = None
data_input_override = None
output_prefix = ""
dataset_year_override = None
train_tags = []

i = 0
while i < len(args):
    token = args[i]
    if token == "--output-tag" and i + 1 < len(args):
        output_tag = args[i + 1]
        i += 2
        continue
    if token == "--data-input" and i + 1 < len(args):
        data_input_override = args[i + 1]
        i += 2
        continue
    if token == "--output-prefix" and i + 1 < len(args):
        output_prefix = args[i + 1]
        i += 2
        continue
    if token == "--dataset-year" and i + 1 < len(args):
        dataset_year_override = args[i + 1]
        i += 2
        continue
    train_tags.append(token)
    i += 1

group_tag = train_group_tag(train_tags)
output_tag = output_tag or group_tag

APPLY_DATASET_OPTIONS = {
    "2023": {
        "mc_input": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC.root:ntmix",
        "data_input": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root:ntmix",
    },
    "2024": {
        "mc_input": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_MC.root:ntmix",
        "data_input": "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root:ntmix",
    },
}

def infer_default_dataset_year(tag):
    if tag.startswith("pb23"):
        return "2023"
    if tag.startswith("pb24"):
        return "2024"
    return None


dataset_year = dataset_year_override or infer_default_dataset_year(output_tag) or infer_default_dataset_year(group_tag)
if dataset_year is None or dataset_year not in APPLY_DATASET_OPTIONS:
    raise ValueError(
        f"Unable to resolve dataset year for output_tag='{output_tag}' group_tag='{group_tag}'. "
        f"Please pass --dataset-year with one of: {sorted(APPLY_DATASET_OPTIONS)}"
    )
inputs = APPLY_DATASET_OPTIONS[dataset_year]
dataset_source = f"PbPb{dataset_year}"

MC_INPUT = inputs["mc_input"]
DATA_INPUT = data_input_override or inputs["data_input"]

print(f"Apply dataset source: {dataset_source}")
print(f"Apply MC input: {MC_INPUT}")
print(f"Apply DATA input: {DATA_INPUT}")

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
print(f"Writing grouped outputs under: {output_tag}")
models = []
skipped_models = []
reference_input_columns = None
reference_trans_columns = None

for train_tag in train_tags:
    try:
        resolved_model_path = resolve_model_path(train_tag)
        resolved_scaler_path = resolve_scaler_path(train_tag)
        resolved_config_path = resolve_model_config_path(train_tag)

        xgbc = joblib.load(resolved_model_path)
        scaler = joblib.load(resolved_scaler_path)

        with open(resolved_config_path) as f:
            config = json.load(f)
    except Exception as exc:
        skipped_models.append({"train_tag": train_tag, "reason": str(exc)})
        print(f"  [SKIP] {train_tag}: {exc}")
        continue

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

if not models:
    print("No valid models found in this group; nothing to apply.")
    if skipped_models:
        print("Skipped models:")
        for item in skipped_models:
            print(f"  - {item['train_tag']}: {item['reason']}")
    sys.exit(1)

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


output_dir = ensure_dir(selected_dir(output_tag))
print(f"Writing grouped scored events to: {output_dir}")
print(f"Output prefix: '{output_prefix}'")

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

mc_path = os.path.join(output_dir, f"{output_prefix}MC_with_score.root")
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

data_path = os.path.join(output_dir, f"{output_prefix}DATA_with_score.root")
with uproot.recreate(data_path) as f:
    f["ntmix"] = {col: df_data_out[col].values for col in df_data_out.columns}
print(f"  Saved grouped DATA to: {data_path}")

summary_path = os.path.join(output_dir, f"{output_prefix}batch_apply_summary.json")
with open(summary_path, "w") as f:
    json.dump(
        {
            "group_tag": group_tag,
            "output_tag": output_tag,
            "output_prefix": output_prefix,
            "requested_train_tags": train_tags,
            "applied_train_tags": [m["train_tag"] for m in models],
            "skipped_models": skipped_models,
            "data_input": DATA_INPUT,
            "mc_input": MC_INPUT,
            "mc_output": mc_path,
            "data_output": data_path,
        },
        f,
        indent=2,
    )
print(f"Saved apply summary: {summary_path}")

if skipped_models:
    print("Skipped models:")
    for item in skipped_models:
        print(f"  - {item['train_tag']}: {item['reason']}")

print(f"\nAll grouped outputs saved in: {output_dir}")
