import os
import re
import sys
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_draw_config,
    resolve_fiducial_config,
    split_root_spec,
)
from utils.paths import ensure_dir, selected_dir, train_group_tag
from utils.paths import resolve_model_config_path
from utils.selection import apply_selection, selection_columns
from utils.score_thresholds import weighted_efficiency_thresholds

if len(sys.argv) < 2:
    print(
        "Usage: python3 workflows/batch_draw_scores.py "
        "[--output-tag <output_tag>] [--output-prefix <prefix>] [--fid-profile <auto|fid{n}>] "
        "[--draw-config <json_path>] "
        "<train_tag> [<train_tag> ...]"
    )
    sys.exit(1)

args = sys.argv[1:]
output_tag = None
output_prefix = ""
fid_profile = "auto"
draw_config_path = None
train_tags = []

i = 0
while i < len(args):
    token = args[i]
    if token == "--output-tag" and i + 1 < len(args):
        output_tag = args[i + 1]
        i += 2
        continue
    if token == "--output-prefix" and i + 1 < len(args):
        output_prefix = args[i + 1]
        i += 2
        continue
    if token == "--fid-profile" and i + 1 < len(args):
        fid_profile = args[i + 1]
        i += 2
        continue
    if token == "--draw-config" and i + 1 < len(args):
        draw_config_path = args[i + 1]
        i += 2
        continue
    train_tags.append(token)
    i += 1

if not (fid_profile == "auto" or re.fullmatch(r"fid\d*", fid_profile)):
    raise ValueError("--fid-profile must be 'auto' or match fid{n} (for example: fid, fid2, fid3)")

group_tag = train_group_tag(train_tags)
output_tag = output_tag or group_tag
sample_key = infer_sample_from_tag(output_tag)
channel = infer_channel_from_tag(output_tag)
dataset_year = infer_dataset_year(output_tag, sample_key)
active_fid = infer_fid_profile(output_tag, sample_key) if fid_profile == "auto" else fid_profile
fid_cfg = resolve_fiducial_config(sample_key, channel, active_fid)
draw_cfg = resolve_draw_config(sample_key, channel, dataset_year)

TREE = draw_cfg["data"]["tree"]
MASS_RANGE = tuple(draw_cfg["plot"]["mass_range"])
BIN_WIDTH = float(draw_cfg["plot"]["bin_width"])
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
REFERENCE_MASSES = list(draw_cfg["plot"].get("reference_masses", []))
score_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95]
efficiency_targets = [
    0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35,
    0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
]


def format_plot_title(train_tag, detail):
    wrapped_tag = "\n".join(
        textwrap.wrap(
            train_tag,
            width=48,
            break_long_words=True,
            break_on_hyphens=False,
        )
    )
    return f"{wrapped_tag}\n{detail}"

if draw_config_path:
    with open(draw_config_path) as f:
        draw_cfg_map = json.load(f)
    tag_cfg = draw_cfg_map.get(output_tag)
    if tag_cfg is None:
        raise ValueError(
            f"Tag '{output_tag}' not found in draw config '{draw_config_path}'."
        )
    if "score_cuts" in tag_cfg:
        cuts = tag_cfg["score_cuts"]
        if not isinstance(cuts, list) or not cuts:
            raise ValueError(f"Invalid score_cuts for tag '{output_tag}' in '{draw_config_path}'")
        score_cuts = [float(x) for x in cuts]
    if "mass_range" in tag_cfg:
        mass_range = tag_cfg["mass_range"]
        if not isinstance(mass_range, list) or len(mass_range) != 2:
            raise ValueError(f"Invalid mass_range for tag '{output_tag}' in '{draw_config_path}'")
        MASS_RANGE = (float(mass_range[0]), float(mass_range[1]))
    if "bin_width" in tag_cfg:
        BIN_WIDTH = float(tag_cfg["bin_width"])
    if BIN_WIDTH <= 0:
        raise ValueError(f"bin_width must be > 0 for tag '{output_tag}'")
    if MASS_RANGE[1] <= MASS_RANGE[0]:
        raise ValueError(f"mass_range must satisfy min < max for tag '{output_tag}'")
    BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    print(f"Using draw overrides from {draw_config_path} for tag {output_tag}")
    print(f"  score_cuts={score_cuts}")
    print(f"  mass_range={MASS_RANGE}")
    print(f"  bin_width={BIN_WIDTH}")

input_file = os.path.join(selected_dir(output_tag), f"{output_prefix}DATA_with_score.root")
if not os.path.exists(input_file):
    print(f"Grouped DATA file not found: {input_file}")
    sys.exit(1)

root_file = uproot.open(input_file)
tree = root_file[TREE]
available_branches = set(tree.keys())
score_branch_map = {}
valid_train_tags, missing_train_tags = [], []
for train_tag in train_tags:
    tagged_branch = f"Prediction_{train_tag}"
    if tagged_branch in available_branches:
        score_branch_map[train_tag] = tagged_branch
        valid_train_tags.append(train_tag)
    elif "Prediction" in available_branches:
        score_branch_map[train_tag] = "Prediction"
        valid_train_tags.append(train_tag)
    else:
        missing_train_tags.append(train_tag)

if not valid_train_tags:
    print("No valid score branches found.")
    sys.exit(1)

draw_columns = list(
    dict.fromkeys(
        ["Bmass"]
        + list(score_branch_map.values())
        + selection_columns(fid_cfg.get("expression"))
    )
)
missing_draw_columns = [
    column for column in draw_columns if column not in available_branches
]
if missing_draw_columns:
    raise ValueError(
        f"Draw input is missing columns required by mass/score/fiducial selection: "
        f"{missing_draw_columns}"
    )
print(f"Draw read columns ({len(draw_columns)}): {draw_columns}")
df = tree.arrays(draw_columns, library="pd")

df_base = df[(df["Bmass"] > MASS_RANGE[0]) & (df["Bmass"] < MASS_RANGE[1])]
df_fid = apply_selection(df_base, fid_cfg.get("expression"), f"fiducial_profiles[{active_fid}]")

# Significance scan is intentionally disabled for now.
for train_tag in valid_train_tags:
    score_column = score_branch_map[train_tag]
    cut_scan_root = ensure_dir(os.path.join(selected_dir(output_tag), "cut_scan"))
    score_cut_root = ensure_dir(os.path.join(cut_scan_root, "score_cut"))
    if len(valid_train_tags) == 1:
        output_dir = score_cut_root
    else:
        output_dir = ensure_dir(os.path.join(cut_scan_root, f"{output_prefix}{train_tag}"))
    for cut in score_cuts:
        cut_tag = int(round(cut * 1000)) if cut >= 0.99 else int(round(cut * 100))
        df_cut = df_fid[df_fid[score_column] > cut]

        plt.figure(figsize=(6, 6))
        plt.hist(df_cut["Bmass"], bins=BINS, histtype="step", linewidth=2)
        for mass in REFERENCE_MASSES:
            plt.axvline(mass, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
        plt.xlabel("Bmass (GeV)")
        plt.ylabel("Entries")
        plt.title(format_plot_title(train_tag, f"score > {cut}"), fontsize=10)
        plt.xlim(MASS_RANGE[0], MASS_RANGE[1])
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if cut >= 0.99:
            out_name = os.path.join(output_dir, f"DATA_{active_fid}_cut{cut_tag:04d}.pdf")
        else:
            out_name = os.path.join(output_dir, f"DATA_{active_fid}_cut{cut_tag:03d}.pdf")
        plt.savefig(out_name, bbox_inches="tight")
        plt.close()

    reference_file = os.path.join(selected_dir(output_tag), f"{output_prefix}REFERENCE_MC_with_score.root")
    if os.path.exists(reference_file):
        with open(resolve_model_config_path(train_tag)) as f:
            model_cfg = json.load(f)
        weight_branch = model_cfg.get("efficiency_reference_weight_branch")
        efficiency_label = "weighted signal efficiency" if weight_branch else "signal efficiency"
        reference_tree_name = split_root_spec(model_cfg["efficiency_reference_signal"])[1]
        reference_tree = uproot.open(reference_file)[reference_tree_name]
        ref_columns = list(dict.fromkeys([score_column] + ([weight_branch] if weight_branch else []) + selection_columns(fid_cfg.get("expression"))))
        ref_df = reference_tree.arrays(ref_columns, library="pd")
        ref_df = apply_selection(ref_df, fid_cfg.get("expression"), f"fiducial_profiles[{active_fid}]")
        scores = ref_df[score_column].to_numpy(dtype=float)
        weights = ref_df[weight_branch].to_numpy(dtype=float) if weight_branch else np.ones(len(ref_df))
        valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
        scores, weights = scores[valid], weights[valid]
        threshold_rows = weighted_efficiency_thresholds(
            scores, weights, efficiency_targets
        )
        efficiency_dir = ensure_dir(os.path.join(cut_scan_root, "weighted_signal_efficiency"))
        efficiency_rows = []
        for row in threshold_rows:
            target = row["target_efficiency"]
            threshold = row["score_threshold"]
            achieved = row["achieved_efficiency"]
            data_cut = df_fid[df_fid[score_column] > threshold]
            plt.figure(figsize=(6, 6))
            plt.hist(data_cut["Bmass"], bins=BINS, histtype="step", linewidth=2)
            for mass in REFERENCE_MASSES:
                plt.axvline(mass, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
            plt.xlabel("Bmass (GeV)"); plt.ylabel("Entries")
            plt.title(
                format_plot_title(train_tag, f"{efficiency_label} {target:.0%}"),
                fontsize=10,
            )
            plt.xlim(MASS_RANGE[0], MASS_RANGE[1]); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(
                os.path.join(efficiency_dir, f"DATA_{active_fid}_eff{int(target*100):03d}.pdf"),
                bbox_inches="tight",
            ); plt.close()
            efficiency_rows.append({"target_efficiency": target, "score_threshold": threshold, "achieved_efficiency": achieved, "data_entries": int(len(data_cut))})
        with open(os.path.join(efficiency_dir, "thresholds.json"), "w") as f:
            json.dump({"train_tag": train_tag, "efficiency_label": efficiency_label, "weight_branch": weight_branch, "reference_file": reference_file, "thresholds": efficiency_rows}, f, indent=2)
