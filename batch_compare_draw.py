import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from utils.paths import cut_scan_dir, ensure_dir, resolve_model_config_path, resolve_model_path, resolve_scaler_path, train_group_tag

if len(sys.argv) < 2:
    print("Usage: python3 batch_compare_draw.py <train_tag> [<train_tag> ...]")
    sys.exit(1)

train_tags = sys.argv[1:]
group_tag = train_group_tag(train_tags)

DATA_INPUT = "/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA.root:ntmix"
TREE = "ntmix"
MASS_RANGE = (3.62, 4.0)
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.01)
BQVALUE_MAX = 0.13
score_cuts = [0.0, 0.5, 0.6, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.99, 0.993, 0.995]


def ordered_unique(columns):
    seen = set()
    output = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


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
            "score_column": f"xgb_score_{train_tag}",
        }
    )

input_columns = reference_input_columns
trans_columns = reference_trans_columns
branches = ordered_unique(["Bmass", "BQvalue"] + input_columns)

print(f"\nLoading DATA once: {DATA_INPUT}")
df = uproot.concatenate(DATA_INPUT, filter_name=branches, library="pd")
print(f"  Loaded {len(df)} events with branches: {branches}")

df = df[
    (df["Bmass"] > MASS_RANGE[0])
    & (df["Bmass"] < MASS_RANGE[1])
    & (df["BQvalue"] < BQVALUE_MAX)
].copy()
print(f"  After mass + BQvalue<{BQVALUE_MAX}: {len(df)}")

def scaler_cache_key(model_bundle):
    scaler = model_bundle["scaler"]
    return (
        tuple(model_bundle["input_columns"]) if "input_columns" in model_bundle else tuple(input_columns),
        tuple(model_bundle["trans_columns"]) if "trans_columns" in model_bundle else tuple(trans_columns),
        tuple(np.asarray(scaler.mean_, dtype=float).round(12)),
        tuple(np.asarray(scaler.scale_, dtype=float).round(12)),
    )


transform_cache = {}

for model_bundle in models:
    train_tag = model_bundle["train_tag"]
    score_column = model_bundle["score_column"]
    output_dir = ensure_dir(cut_scan_dir(train_tag))
    cache_key = scaler_cache_key(model_bundle)
    if cache_key not in transform_cache:
        transform_cache[cache_key] = pd.DataFrame(
            model_bundle["scaler"].transform(df[input_columns]),
            columns=trans_columns,
            index=df.index,
        )
    df_trans = transform_cache[cache_key]
    scores = model_bundle["model"].predict_proba(df_trans[trans_columns])[:, 1]
    df[score_column] = scores
    print(
        f"  [{train_tag}] Score range for {score_column}: "
        f"[{scores.min():.4f}, {scores.max():.4f}]"
    )

    for cut in score_cuts:
        df_cut = df[df[score_column] > cut]

        plt.figure(figsize=(6, 6))
        plt.hist(
            df_cut["Bmass"],
            bins=BINS,
            histtype="step",
            linewidth=2,
        )

        n_entries = len(df_cut)
        mean = df_cut["Bmass"].mean()
        std = df_cut["Bmass"].std()
        textstr = "\n".join(
            (
                f"Entries = {n_entries}",
                f"Mean = {mean:.4f}",
                f"Std Dev = {std:.4f}",
                f"BQvalue < {BQVALUE_MAX}",
            )
        )

        plt.text(
            0.97,
            0.97,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.xlabel("Bmass (GeV)")
        plt.ylabel("Entries")
        plt.title(f"{train_tag} | score > {cut}")
        plt.xlim(MASS_RANGE[0], MASS_RANGE[1])
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if cut >= 0.99:
            cut_tag = int(round(cut * 1000))
            out_name = f"{output_dir}/DATA_cut{cut_tag:04d}.pdf"
        else:
            cut_tag = int(round(cut * 100))
            out_name = f"{output_dir}/DATA_cut{cut_tag:03d}.pdf"

        plt.savefig(out_name)
        plt.close()
        print(f"Saved: {out_name}")

    del df[score_column]

print("\nAll plots saved.")
