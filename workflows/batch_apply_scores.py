import json
import os
import sys

import joblib
import pandas as pd
import uproot

from configs.samples import (
    bd_pbpb_precut_paths,
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_apply_config,
    resolve_draw_config,
    resolve_extra_mc_apply_config,
    resolve_fiducial_config,
    resolve_training_config,
    split_root_spec,
    supports_bd_pbpb_precut,
    to_root_spec,
)
from utils.paths import ensure_dir, resolve_model_config_path, resolve_model_path, resolve_scaler_path, selected_dir, train_group_tag
from utils.varsets import get_varset_columns, infer_varset_from_tag

if len(sys.argv) < 2:
    print(
        "Usage: python3 workflows/batch_apply_scores.py "
        "[--output-tag <output_tag>] [--data-input <root:tree>] [--output-prefix <prefix>] [--dataset-year <year>] [--use-precut <0|1>] [--apply-extra-mc <keys|all>] "
        "<train_tag> [<train_tag> ...]"
    )
    sys.exit(1)

args = sys.argv[1:]
output_tag = None
data_input_override = None
output_prefix = ""
dataset_year_override = None
use_precut = 0
apply_extra_mc = None
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
    if token == "--use-precut" and i + 1 < len(args):
        use_precut = int(args[i + 1])
        i += 2
        continue
    if token == "--apply-extra-mc" and i + 1 < len(args):
        apply_extra_mc = args[i + 1]
        i += 2
        continue
    train_tags.append(token)
    i += 1

group_tag = train_group_tag(train_tags)
output_tag = output_tag or group_tag
sample_key = infer_sample_from_tag(output_tag)
channel = infer_channel_from_tag(output_tag)
dataset_year = dataset_year_override or infer_dataset_year(output_tag, sample_key)
apply_cfg = resolve_apply_config(sample_key, channel, dataset_year)
draw_cfg = resolve_draw_config(sample_key, channel, dataset_year)
draw_tree_name = draw_cfg["data"]["tree"]

MC_INPUT = to_root_spec(apply_cfg["mc"][0])
DATA_INPUT = data_input_override or to_root_spec(apply_cfg["data"][0])
MC_TREE_INPUT = split_root_spec(MC_INPUT)[1]

if use_precut:
    if not supports_bd_pbpb_precut(output_tag):
        raise ValueError(f"--use-precut only supports Bd_pb23/Bd_pb24 single-DAG tags, got '{output_tag}'.")
    if data_input_override:
        raise ValueError("--use-precut cannot be combined with --data-input.")
    precut_paths = bd_pbpb_precut_paths(output_tag)
    apply_data_path = precut_paths["apply_data"]
    if not apply_data_path.exists():
        raise FileNotFoundError(f"Missing precut apply DATA file: {apply_data_path}")
    _, data_tree = split_root_spec(DATA_INPUT)
    DATA_INPUT = f"{apply_data_path}:{data_tree}"

print(f"Apply dataset source: {apply_cfg['dataset_source']}")
print(f"Apply MC input: {MC_INPUT}")
print(f"Apply DATA input: {DATA_INPUT}")

single_model_mode = len(train_tags) == 1


def score_branch_name(train_tag):
    if single_model_mode:
        return "Prediction"
    return f"Prediction_{train_tag}"


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
    sys.exit(1)

input_columns = reference_input_columns
trans_columns = reference_trans_columns


def score_dataframe(df, model_bundle):
    df_trans = pd.DataFrame(
        model_bundle["scaler"].transform(df[model_bundle["input_columns"]]),
        columns=model_bundle["trans_columns"],
        index=df.index,
    )
    scores = model_bundle["model"].predict_proba(df_trans[model_bundle["trans_columns"]])[:, 1]
    df_out = df.copy()
    df_out[model_bundle["score_column"]] = scores
    return df_out


output_dir = ensure_dir(selected_dir(output_tag))

if apply_extra_mc is not None:
    if channel != "X":
        raise ValueError("--apply-extra-mc only supports X channel.")
    if use_precut:
        raise ValueError("--apply-extra-mc cannot be combined with --use-precut.")
    extra_cfg = resolve_extra_mc_apply_config(sample_key, channel, dataset_year, apply_extra_mc)
    failures = []
    for key, spec in extra_cfg["samples"].items():
        EXTRA_INPUT = to_root_spec(spec)
        extra_tree = split_root_spec(EXTRA_INPUT)[1]
        out_path = os.path.join(output_dir, f"{output_prefix}MC_{key}_with_score.root")
        print(f"Processing extra MC [{key}]: {EXTRA_INPUT}")
        try:
            df_extra = uproot.concatenate(EXTRA_INPUT, library="pd")
            df_extra_out = None
            for model_bundle in models:
                df_scored = score_dataframe(df_extra, model_bundle)
                if df_extra_out is None:
                    df_extra_out = df_scored
                else:
                    df_extra_out[model_bundle["score_column"]] = df_scored[model_bundle["score_column"]]
            with uproot.recreate(out_path) as f:
                f[extra_tree] = {col: df_extra_out[col].values for col in df_extra_out.columns}
            print(f"Saved extra MC with score: {out_path}")
        except Exception as exc:
            failures.append({"key": key, "input": EXTRA_INPUT, "reason": str(exc)})
            print(f"  [SKIP] extra MC {key}: {exc}")
    if failures:
        print(f"Completed extra MC apply with {len(failures)} failure(s): {[f['key'] for f in failures]}")
    sys.exit(0)

print(f"Processing MC: {MC_INPUT}")
df_mc = uproot.concatenate(MC_INPUT, library="pd")
df_mc_out = None
for model_bundle in models:
    df_scored = score_dataframe(df_mc, model_bundle)
    if df_mc_out is None:
        df_mc_out = df_scored
    else:
        df_mc_out[model_bundle["score_column"]] = df_scored[model_bundle["score_column"]]

mc_path = os.path.join(output_dir, f"{output_prefix}MC_with_score.root")
with uproot.recreate(mc_path) as f:
    f[MC_TREE_INPUT] = {col: df_mc_out[col].values for col in df_mc_out.columns}

print(f"Processing DATA: {DATA_INPUT}")
df_data = uproot.concatenate(DATA_INPUT, library="pd")
df_data_out = None
for model_bundle in models:
    df_scored = score_dataframe(df_data, model_bundle)
    if df_data_out is None:
        df_data_out = df_scored
    else:
        df_data_out[model_bundle["score_column"]] = df_scored[model_bundle["score_column"]]

data_path = os.path.join(output_dir, f"{output_prefix}DATA_with_score.root")
with uproot.recreate(data_path) as f:
    f[draw_tree_name] = {col: df_data_out[col].values for col in df_data_out.columns}

summary_path = os.path.join(output_dir, f"{output_prefix}batch_apply_summary.json")
selection_profile = infer_selection_profile(output_tag, sample_key)
training_cfg = resolve_training_config(sample_key, channel, dataset_year, selection_profile)
fid_profile = infer_fid_profile(output_tag, sample_key)
fid_cfg = resolve_fiducial_config(sample_key, channel, fid_profile)

varset_tag = None
varset_columns = []
for tag in [m["train_tag"] for m in models] + train_tags:
    candidate = infer_varset_from_tag(tag, sample=sample_key)
    if candidate is not None:
        varset_tag = candidate
        varset_columns = get_varset_columns(sample_key, candidate, channel=channel)
        break

with open(summary_path, "w") as f:
    json.dump(
        {
            "input_datasets": {
                "sample": sample_key,
                "channel": channel,
                "dataset_year": dataset_year,
                "mc_input": MC_INPUT,
                "data_input": DATA_INPUT,
            },
            "input_selection": {
                "selection_profile": selection_profile,
                "signal_selection": training_cfg["signal_selection"],
                "background_selection": training_cfg["background_selection"],
            },
            "draw_selection": {
                "fid_profile": fid_profile,
                "fiducial_cut": fid_cfg,
            },
            "training_varset": {
                "varset_tag": varset_tag,
                "columns": varset_columns,
                "train_tags": [m["train_tag"] for m in models],
            },
        },
        f,
        indent=2,
    )
print(f"Saved apply summary: {summary_path}")
